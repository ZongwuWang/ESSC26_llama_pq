// PQ decode-time GEMV kernels + tensor registry. See pq-gemv.h.
// Kernels are ith/nth-parallel ports of pq_cpu/bench_pq.cpp s1_gemv_i8lut /
// s2_gemv_i8lut (AVX-512 FP16 + VBMI, Granite Rapids target).
#include "ggml-pq.h"

#include <immintrin.h>
#include <sys/mman.h>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <map>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "ggml-cpu-impl.h" // ggml_barrier
#include "ggml.h"

// Kernels require AVX-512 FP16 + VBMI; on other targets the registry stays
// empty and every call no-ops (prefill/original path unaffected).
#if defined(__AVX512FP16__) && defined(__AVX512VBMI__) && defined(__AVX512F__)

namespace {

static bool pq_stripe_enabled() {
    const char * value = getenv("GGML_PQ_STRIPE");
    static const bool enabled = value && atoi(value) != 0;
    return enabled;
}

static void pq_pin_cpu_half(int half) {
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 2) return;
    const int begin = half == 0 ? 0 : (int)(ncpu / 2);
    const int end = half == 0 ? (int)(ncpu / 2) : (int)ncpu;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu = begin; cpu < end; cpu++) CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void pq_copy_striped(void * dst, const void * src, size_t bytes) {
    const size_t split = bytes / 2;
    std::thread t0([=] {
        pq_pin_cpu_half(0);
        memcpy(dst, src, split);
    });
    std::thread t1([=] {
        pq_pin_cpu_half(1);
        memcpy((char *) dst + split, (const char *) src + split, bytes - split);
    });
    t0.join();
    t1.join();
}

// Huge-page backed storage for the GEMV streaming buffers (codebooks/indices).
// The per-token weight stream is DRAM-bound; 2MB pages cut TLB misses on the
// multi-GB index streams by ~2 orders of magnitude (see pq_cpu bench: GEMV
// -25~-40% at DRAM scale).
template<typename T>
struct huge_vec {
    T * ptr = nullptr;
    size_t n = 0;
    void free_mem() {
        if (ptr) {
            const size_t HP = 2 * 1024 * 1024;
            munmap(ptr, ((n * sizeof(T) + HP - 1) / HP) * HP);
            ptr = nullptr;
        }
        n = 0;
    }
    void assign(const T * src, size_t count) {
        free_mem();
        n = count;
        const size_t HP = 2 * 1024 * 1024;
        const size_t bytes = ((n * sizeof(T) + HP - 1) / HP) * HP;
        // over-allocate by HP: the 2MB-aligned pointer may sit up to HP-1
        // above raw, so the aligned region stays inside the mapping
        void * raw = mmap(nullptr, bytes + HP, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) { fprintf(stderr, "pq: mmap failed\n"); exit(1); }
        ptr = (T *)(((uintptr_t)raw + HP - 1) & ~(uintptr_t)(HP - 1));
        madvise((void *)ptr, bytes, MADV_HUGEPAGE);
        if (pq_stripe_enabled() && bytes >= 2 * 1024 * 1024) {
            pq_copy_striped(ptr, src, n * sizeof(T));
        } else {
            memcpy(ptr, src, n * sizeof(T));
        }
    }
    huge_vec() = default;
    huge_vec(huge_vec && o) : ptr(o.ptr), n(o.n) { o.ptr = nullptr; o.n = 0; }
    huge_vec & operator=(huge_vec && o) {
        if (this != &o) { free_mem(); ptr = o.ptr; n = o.n; o.ptr = nullptr; o.n = 0; }
        return *this;
    }
    T * data() { return ptr; }
    const T * data() const { return ptr; }
    size_t size() const { return n; }
    ~huge_vec() { free_mem(); }
};

struct PQTensor {
    std::string name;   // registry key (per-tensor timing attribution)
    int      mode = 0;   // 0 = S1, 1 = S2, 2 = fused group of S1 members
    int      ds = 2;
    int64_t  n_in = 0, n_out = 0;
    // mode 2: members run as one S1 GEMV; member m writes dst + seg_off[m]
    int             n_seg = 0;
    const PQTensor * seg_t[4];
    int64_t          seg_off[4];
    // S1
    huge_vec<_Float16> cbh;   // [(i*ds+d)*K + k]
    // S2
    huge_vec<int8_t>   cb8;   // [(i*ds+s)*K + k]
    huge_vec<int8_t>   cb8u;  // same, biased +128 per byte (VNNI unsigned LUT)
    std::vector<float> inv;   // [i*ds+s], pre-scaled by 128 (small)
    huge_vec<uint8_t>  idx;   // S1: [i*n_out + j]; S2: [i*n_in + j]
    std::vector<float> row_scale;
    bool scaled = false;
};

std::unordered_map<std::string, PQTensor> g_pq;
std::atomic<bool> g_pq_enabled{false};

constexpr int kMaxThreads = 256;

// shared fp16 partial-y pool for the S1 two-phase scheme
std::vector<_Float16> g_s1_partials;
std::mutex            g_s1_mu;

// per-tensor MUL_MAT busy-time aggregation (top-list printed by dump)
struct MulmatAcc { uint64_t busy_us = 0, count = 0; };
static std::map<std::string, MulmatAcc> g_mulmat_acc;
static std::mutex g_mulmat_mu;

static inline __m512i load_i8_lut256(const int8_t * t) {
    return _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)t)),
        _mm256_loadu_si256((const __m256i *)(t + 32)), 1);
}

// 256-entry int8 LUT from 4 zmm byte tables; vpermi2b zeroes out-of-range
// indices, so 2x vpermi2b + 1 blend cover entries 0..255.
static inline __m512i lut256_u8(__m512i idx, const __m512i t[4]) {
    const __m512i q0 = _mm512_permutex2var_epi8(t[0], idx, t[1]);
    const __m512i q1 = _mm512_permutex2var_epi8(t[2],
        _mm512_xor_si512(idx, _mm512_set1_epi8((char)0x80)), t[3]);
    return _mm512_mask_blend_epi8(
        _mm512_test_epi8_mask(idx, _mm512_set1_epi8((char)0x80)), q0, q1);
}

static inline __m512 fp16_low_to_fp32(__m512i v) {
    return _mm512_cvtph_ps(_mm512_castsi512_si256(v));
}
static inline __m512 fp16_high_to_fp32(__m512i v) {
    return _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v, 1));
}

struct S2XBufs { const _Float16 * xh; const int8_t * xq; float xs; int64_t sx; };

// ---------------------------------------------------------------------------
// S2: y[i*ds+s] = sum_j x[j] * c_s(idx_ij) * inv[i*ds+s]
// x arrives pre-converted to fp16 and scaled by 1/128 (xh); the VNNI path
// additionally uses the int8-quantized x (xq, xs) and the biased codebook.
// ---------------------------------------------------------------------------
static void pq_s2_gemv(const PQTensor & t, const S2XBufs & xb, float * dst,
                       int ith, int nth) {
    const int ds = t.ds;
    const int M  = (int)(t.n_out / ds);
    const int din = (int)t.n_in;
    const int8_t  * cb8 = t.cb8.data();
    const int8_t  * cb8u = t.cb8u.data();
    const float   * inv = t.inv.data();
    const uint8_t * idx = t.idx.data();
    const _Float16 * xh = xb.xh;
    const int8_t  * xq = xb.xq;
    const float xs = xb.xs;
    const int64_t sx = xb.sx;

    const int i0 = (int)((int64_t)M * ith / nth);
    const int i1 = (int)((int64_t)M * (ith + 1) / nth);

    if (ds == 2) {
        // VNNI path: LUT codes (unsigned, from the +128-biased codebook) feed
        // vpdpbusd (1 uop / 64 int8 MACs, int32 accumulation); the signed
        // contribution is recovered via  sum(c*xh) = (acc_u - 128*sx)/xs.
        for (int i = i0; i < i1; i++) {
            const int8_t * ceu = cb8u + (size_t)i * 512;
            const int8_t * cou = ceu + 256;
            const int8_t * ce = cb8 + (size_t)i * 512;
            const int8_t * co = ce + 256;
            const uint8_t * ii = idx + (size_t)i * din;
            if (i + 1 < i1) {
                _mm_prefetch((const char *)(idx + (size_t)(i + 1) * din), _MM_HINT_T0);
                _mm_prefetch((const char *)(cb8u + (size_t)(i + 1) * 512), _MM_HINT_T0);
            }
            // deep prefetch: cover DRAM latency, keep more subspaces in flight
            if (i + 8 < i1) {
                _mm_prefetch((const char *)(idx + (size_t)(i + 8) * din), _MM_HINT_T1);
                _mm_prefetch((const char *)(cb8u + (size_t)(i + 8) * 512), _MM_HINT_T1);
            }
            if (i + 16 < i1) {
                _mm_prefetch((const char *)(idx + (size_t)(i + 16) * din), _MM_HINT_T1);
                _mm_prefetch((const char *)(cb8u + (size_t)(i + 16) * 512), _MM_HINT_T1);
            }
            __m512i te[4], to[4];
            for (int q = 0; q < 4; q++) {
                te[q] = load_i8_lut256(ceu + q * 64);
                to[q] = load_i8_lut256(cou + q * 64);
            }
            __m512i ae0 = _mm512_setzero_si512(), ae1 = _mm512_setzero_si512();
            __m512i ao0 = _mm512_setzero_si512(), ao1 = _mm512_setzero_si512();
            __m512i ae2 = _mm512_setzero_si512(), ae3 = _mm512_setzero_si512();
            __m512i ao2 = _mm512_setzero_si512(), ao3 = _mm512_setzero_si512();
            int j = 0;
            for (; j + 255 < din; j += 256) {
                const __m512i idx0 = _mm512_loadu_si512((const void *)(ii + j));
                const __m512i idx1 = _mm512_loadu_si512((const void *)(ii + j + 64));
                const __m512i idx2 = _mm512_loadu_si512((const void *)(ii + j + 128));
                const __m512i idx3 = _mm512_loadu_si512((const void *)(ii + j + 192));
                const __m512i qe0 = lut256_u8(idx0, te);
                const __m512i qo0 = lut256_u8(idx0, to);
                const __m512i qe1 = lut256_u8(idx1, te);
                const __m512i qo1 = lut256_u8(idx1, to);
                const __m512i qe2 = lut256_u8(idx2, te);
                const __m512i qo2 = lut256_u8(idx2, to);
                const __m512i qe3 = lut256_u8(idx3, te);
                const __m512i qo3 = lut256_u8(idx3, to);
                const __m512i xq0 = _mm512_loadu_si512((const void *)(xq + j));
                const __m512i xq1 = _mm512_loadu_si512((const void *)(xq + j + 64));
                const __m512i xq2 = _mm512_loadu_si512((const void *)(xq + j + 128));
                const __m512i xq3 = _mm512_loadu_si512((const void *)(xq + j + 192));
                ae0 = _mm512_dpbusd_epi32(ae0, qe0, xq0);
                ao0 = _mm512_dpbusd_epi32(ao0, qo0, xq0);
                ae1 = _mm512_dpbusd_epi32(ae1, qe1, xq1);
                ao1 = _mm512_dpbusd_epi32(ao1, qo1, xq1);
                ae2 = _mm512_dpbusd_epi32(ae2, qe2, xq2);
                ao2 = _mm512_dpbusd_epi32(ao2, qo2, xq2);
                ae3 = _mm512_dpbusd_epi32(ae3, qe3, xq3);
                ao3 = _mm512_dpbusd_epi32(ao3, qo3, xq3);
            }
            for (; j + 63 < din; j += 64) {
                const __m512i idxv = _mm512_loadu_si512((const void *)(ii + j));
                const __m512i qe = lut256_u8(idxv, te);
                const __m512i qo = lut256_u8(idxv, to);
                const __m512i xqv = _mm512_loadu_si512((const void *)(xq + j));
                ae0 = _mm512_dpbusd_epi32(ae0, qe, xqv);
                ao0 = _mm512_dpbusd_epi32(ao0, qo, xqv);
            }
            const float corr = 128.0f * (float)sx;
            const __m512i ae = _mm512_add_epi32(_mm512_add_epi32(ae0, ae1),
                                                _mm512_add_epi32(ae2, ae3));
            const __m512i ao = _mm512_add_epi32(_mm512_add_epi32(ao0, ao1),
                                                _mm512_add_epi32(ao2, ao3));
            float s0 = ((float)_mm512_reduce_add_epi32(ae) - corr) / xs;
            float s1 = ((float)_mm512_reduce_add_epi32(ao) - corr) / xs;
            for (; j < din; j++) {
                s0 += (float)xh[j] * ce[ii[j]];
                s1 += (float)xh[j] * co[ii[j]];
            }
            dst[(size_t)i * 2]     = s0 * inv[(size_t)i * 2];
            dst[(size_t)i * 2 + 1] = s1 * inv[(size_t)i * 2 + 1];
        }
        return;
    }


    // general ds <= 8: idx kept in registers across sides, per-side tables
    // streamed from L1.
    for (int i = i0; i < i1; i++) {
        const int8_t * cbi = cb8 + (size_t)i * ds * GGML_PQ_K;
        const uint8_t * ii = idx + (size_t)i * din;
        if (i + 1 < i1) {
            _mm_prefetch((const char *)(idx + (size_t)(i + 1) * din), _MM_HINT_T0);
            _mm_prefetch((const char *)(cb8 + (size_t)(i + 1) * ds * GGML_PQ_K), _MM_HINT_T0);
        }
        // deep prefetch: cover DRAM latency, keep more subspaces in flight
        if (i + 8 < i1) {
            _mm_prefetch((const char *)(idx + (size_t)(i + 8) * din), _MM_HINT_T1);
            _mm_prefetch((const char *)(cb8 + (size_t)(i + 8) * ds * GGML_PQ_K), _MM_HINT_T1);
        }
        if (i + 16 < i1) {
            _mm_prefetch((const char *)(idx + (size_t)(i + 16) * din), _MM_HINT_T1);
            _mm_prefetch((const char *)(cb8 + (size_t)(i + 16) * ds * GGML_PQ_K), _MM_HINT_T1);
        }
        __m512h acc[GGML_PQ_MAX_DS][2];
        for (int s = 0; s < ds; s++) {
            acc[s][0] = _mm512_setzero_ph();
            acc[s][1] = _mm512_setzero_ph();
        }
        int j = 0;
        for (; j + 63 < din; j += 64) {
            const __m512i idxv = _mm512_loadu_si512((const void *)(ii + j));
            const __m512h xe0 = _mm512_loadu_ph(xh + j);
            const __m512h xe1 = _mm512_loadu_ph(xh + j + 32);
            for (int s = 0; s < ds; s++) {
                const int8_t * cs = cbi + (size_t)s * GGML_PQ_K;
                __m512i t[4];
                t[0] = load_i8_lut256(cs);
                t[1] = load_i8_lut256(cs + 64);
                t[2] = load_i8_lut256(cs + 128);
                t[3] = load_i8_lut256(cs + 192);
                const __m512i q = lut256_u8(idxv, t);
                const __m512h f0 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(q)));
                const __m512h f1 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(q, 1)));
                acc[s][0] = _mm512_fmadd_ph(xe0, f0, acc[s][0]);
                acc[s][1] = _mm512_fmadd_ph(xe1, f1, acc[s][1]);
            }
        }
        float tails[GGML_PQ_MAX_DS] = {0};
        for (int s = 0; s < ds; s++)
            for (int jj = j; jj < din; jj++)
                tails[s] += (float)xh[jj] * cbi[(size_t)s * GGML_PQ_K + ii[jj]];
        for (int s = 0; s < ds; s++) {
            const __m512h e = _mm512_add_ph(acc[s][0], acc[s][1]);
            dst[(size_t)i * ds + s] =
                _mm512_reduce_add_ph(e) * inv[(size_t)i * ds + s] +
                tails[s] * inv[(size_t)i * ds + s];
        }
    }
}

// ---------------------------------------------------------------------------
// S1: two-phase, fusion-capable (matches bench s1_gemv_i8lut).
//   phase 1: threads own disjoint subspace chunks spanning ALL members of a
//            same-input fusion group; per subspace: build fp16 dt table from
//            cbh, int8-quantize, then LUT-accumulate the full output row
//            into this thread's fp16 partial row.
//   barrier
//   phase 2: threads own disjoint output chunks; sum all partials to fp32.
// ---------------------------------------------------------------------------
template<int DS>
static void pq_s1_phase1_impl(const PQTensor & t, const _Float16 * xh,
                              int i0, int i1, _Float16 * yl) {
    constexpr int ds = DS;
    GGML_ASSERT(t.ds == ds);
    const int dout = (int)t.n_out;
    const _Float16 * cbh = t.cbh.data();
    const uint8_t * idx = t.idx.data();

    // Paired-subspace accumulation: subspaces (i, i+1) share one yl
    // read-modify-write per 64B chunk (halves the yl L1 traffic and removes
    // the cross-subspace store-forward chain). dt tables for both subspaces
    // stay in registers; dta/dtb kept for the scalar tail.
    int i = i0;
    for (; i + 1 < i1; i += 2) {
        const _Float16 * cia = cbh + (size_t)i * ds * GGML_PQ_K;
        const _Float16 * cib = cbh + (size_t)(i + 1) * ds * GGML_PQ_K;
        const uint8_t * iia = idx + (size_t)i * dout;
        const uint8_t * iib = idx + (size_t)(i + 1) * dout;
        if (i + 2 < i1) {
            _mm_prefetch((const char *)(idx + (size_t)(i + 2) * dout), _MM_HINT_T0);
            _mm_prefetch((const char *)(cbh + (size_t)(i + 2) * ds * GGML_PQ_K), _MM_HINT_T0);
        }
        if (i + 8 < i1) {
            _mm_prefetch((const char *)(idx + (size_t)(i + 8) * dout), _MM_HINT_T1);
            _mm_prefetch((const char *)(cbh + (size_t)(i + 8) * ds * GGML_PQ_K), _MM_HINT_T1);
        }
        __m512h xba[GGML_PQ_MAX_DS], xbb[GGML_PQ_MAX_DS];
        for (int d = 0; d < ds; d++) {
            xba[d] = _mm512_set1_ph(xh[(size_t)i * ds + d]);
            xbb[d] = _mm512_set1_ph(xh[(size_t)(i + 1) * ds + d]);
        }
        // fp16 dt tables + symmetric int8 quantization for both subspaces
        alignas(64) _Float16 dta[GGML_PQ_K], dtb[GGML_PQ_K];
        __m512h mxa = _mm512_setzero_ph(), mxb = _mm512_setzero_ph();
        for (int k = 0; k < GGML_PQ_K; k += 32) {
            __m512h va = _mm512_mul_ph(xba[0], _mm512_loadu_ph(cia + k));
            __m512h vb = _mm512_mul_ph(xbb[0], _mm512_loadu_ph(cib + k));
            for (int d = 1; d < ds; d++) {
                va = _mm512_fmadd_ph(xba[d],
                    _mm512_loadu_ph(cbh + ((size_t)i * ds + d) * GGML_PQ_K + k), va);
                vb = _mm512_fmadd_ph(xbb[d],
                    _mm512_loadu_ph(cbh + ((size_t)(i + 1) * ds + d) * GGML_PQ_K + k), vb);
            }
            _mm512_storeu_ph(dta + k, va);
            _mm512_storeu_ph(dtb + k, vb);
            mxa = _mm512_max_ph(mxa, _mm512_abs_ph(va));
            mxb = _mm512_max_ph(mxb, _mm512_abs_ph(vb));
        }
        const float amaxa = _mm512_reduce_max_ph(mxa);
        const float amaxb = _mm512_reduce_max_ph(mxb);
        const __m512h iva = _mm512_set1_ph((_Float16)((amaxa > 0.f) ? amaxa / 127.0f : 0.0f));
        const __m512h ivb = _mm512_set1_ph((_Float16)((amaxb > 0.f) ? amaxb / 127.0f : 0.0f));
        const __m512h sca = _mm512_set1_ph((_Float16)(amaxa > 0.f ? 127.0f / amaxa : 0.0f));
        const __m512h scb = _mm512_set1_ph((_Float16)(amaxb > 0.f ? 127.0f / amaxb : 0.0f));

        __m512i t8a[4], t8b[4];
        for (int q = 0; q < 4; q++) {
            const __m512i wa0 = _mm512_cvtph_epi16(_mm512_mul_ph(_mm512_loadu_ph(dta + q * 64), sca));
            const __m512i wa1 = _mm512_cvtph_epi16(_mm512_mul_ph(_mm512_loadu_ph(dta + q * 64 + 32), sca));
            t8a[q] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi16_epi8(wa0)),
                                        _mm512_cvtepi16_epi8(wa1), 1);
            const __m512i wb0 = _mm512_cvtph_epi16(_mm512_mul_ph(_mm512_loadu_ph(dtb + q * 64), scb));
            const __m512i wb1 = _mm512_cvtph_epi16(_mm512_mul_ph(_mm512_loadu_ph(dtb + q * 64 + 32), scb));
            t8b[q] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi16_epi8(wb0)),
                                        _mm512_cvtepi16_epi8(wb1), 1);
        }

        int j = 0;
        for (; j + 63 < dout; j += 64) {
            const __m512i idxa = _mm512_loadu_si512((const void *)(iia + j));
            const __m512i qa0 = _mm512_permutex2var_epi8(t8a[0], idxa, t8a[1]);
            const __m512i qa1 = _mm512_permutex2var_epi8(t8a[2],
                _mm512_xor_si512(idxa, _mm512_set1_epi8((char)0x80)), t8a[3]);
            const __mmask64 ha = _mm512_test_epi8_mask(idxa, _mm512_set1_epi8((char)0x80));
            const __m512i qa = _mm512_mask_blend_epi8(ha, qa0, qa1);
            const __m512h va0 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(qa)));
            const __m512h va1 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(qa, 1)));
            __m512h acc0 = _mm512_loadu_ph(yl + j);
            __m512h acc1 = _mm512_loadu_ph(yl + j + 32);
            acc0 = _mm512_fmadd_ph(va0, iva, acc0);
            acc1 = _mm512_fmadd_ph(va1, iva, acc1);
            const __m512i idxb = _mm512_loadu_si512((const void *)(iib + j));
            const __m512i qb0 = _mm512_permutex2var_epi8(t8b[0], idxb, t8b[1]);
            const __m512i qb1 = _mm512_permutex2var_epi8(t8b[2],
                _mm512_xor_si512(idxb, _mm512_set1_epi8((char)0x80)), t8b[3]);
            const __mmask64 hb = _mm512_test_epi8_mask(idxb, _mm512_set1_epi8((char)0x80));
            const __m512i qb = _mm512_mask_blend_epi8(hb, qb0, qb1);
            const __m512h vb0 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(qb)));
            const __m512h vb1 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(qb, 1)));
            acc0 = _mm512_fmadd_ph(vb0, ivb, acc0);
            acc1 = _mm512_fmadd_ph(vb1, ivb, acc1);
            _mm512_storeu_ph(yl + j, acc0);
            _mm512_storeu_ph(yl + j + 32, acc1);
        }
        for (; j < dout; j++) {
            float acc = (float)yl[j] + (float)dta[iia[j]];
            acc += (float)dtb[iib[j]];
            yl[j] = (_Float16)acc;
        }
    }
    for (; i < i1; i++) {
        const _Float16 * ci = cbh + (size_t)i * ds * GGML_PQ_K;
        const uint8_t * ii = idx + (size_t)i * dout;
        __m512h xb[GGML_PQ_MAX_DS];
        for (int d = 0; d < ds; d++) xb[d] = _mm512_set1_ph(xh[(size_t)i * ds + d]);
        alignas(64) _Float16 dt[GGML_PQ_K];
        __m512h mx = _mm512_setzero_ph();
        for (int k = 0; k < GGML_PQ_K; k += 32) {
            __m512h v = _mm512_mul_ph(xb[0], _mm512_loadu_ph(ci + k));
            for (int d = 1; d < ds; d++)
                v = _mm512_fmadd_ph(xb[d],
                    _mm512_loadu_ph(cbh + ((size_t)i * ds + d) * GGML_PQ_K + k), v);
            _mm512_storeu_ph(dt + k, v);
            mx = _mm512_max_ph(mx, _mm512_abs_ph(v));
        }
        const float amax = _mm512_reduce_max_ph(mx);
        const float inv = (amax > 0.f) ? amax / 127.0f : 0.0f;
        const __m512h iv = _mm512_set1_ph((_Float16)inv);
        const __m512h sc = _mm512_set1_ph((_Float16)(amax > 0.f ? 127.0f / amax : 0.0f));
        __m512i t8[4];
        for (int q = 0; q < 4; q++) {
            const __m512i w0 = _mm512_cvtph_epi16(_mm512_mul_ph(_mm512_loadu_ph(dt + q * 64), sc));
            const __m512i w1 = _mm512_cvtph_epi16(_mm512_mul_ph(_mm512_loadu_ph(dt + q * 64 + 32), sc));
            t8[q] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi16_epi8(w0)),
                                       _mm512_cvtepi16_epi8(w1), 1);
        }
        int j = 0;
        for (; j + 63 < dout; j += 64) {
            const __m512i idxv = _mm512_loadu_si512((const void *)(ii + j));
            const __m512i q0 = _mm512_permutex2var_epi8(t8[0], idxv, t8[1]);
            const __m512i q1 = _mm512_permutex2var_epi8(t8[2],
                _mm512_xor_si512(idxv, _mm512_set1_epi8((char)0x80)), t8[3]);
            const __m512i q = _mm512_mask_blend_epi8(
                _mm512_test_epi8_mask(idxv, _mm512_set1_epi8((char)0x80)), q0, q1);
            const __m512i w0 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(q));
            const __m512i w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(q, 1));
            _mm512_storeu_ph(yl + j,
                _mm512_fmadd_ph(_mm512_cvtepi16_ph(w0), iv, _mm512_loadu_ph(yl + j)));
            _mm512_storeu_ph(yl + j + 32,
                _mm512_fmadd_ph(_mm512_cvtepi16_ph(w1), iv, _mm512_loadu_ph(yl + j + 32)));
        }
        for (; j < dout; j++) yl[j] = (_Float16)((float)yl[j] + dt[ii[j]]);
    }

    // publish partials, then reduce
}

static void pq_s1_phase1_ds4_block(const PQTensor & t, const _Float16 * xh,
                                    int i0, int i1, _Float16 * yl) {
    const int dout = (int) t.n_out;
    const _Float16 * cbh = t.cbh.data();
    const uint8_t * idx = t.idx.data();
    const __m512i sign = _mm512_set1_epi8((char) 0x80);
    int i = i0;
    for (; i + 3 < i1; i += 4) {
        alignas(64) _Float16 dt[4][GGML_PQ_K];
        __m512i lut[4][4];
        __m512h inv[4];
        for (int s = 0; s < 4; s++) {
            const _Float16 * cs = cbh + (size_t)(i + s) * 4 * GGML_PQ_K;
            __m512h x0 = _mm512_set1_ph(xh[(size_t)(i + s) * 4 + 0]);
            __m512h x1 = _mm512_set1_ph(xh[(size_t)(i + s) * 4 + 1]);
            __m512h x2 = _mm512_set1_ph(xh[(size_t)(i + s) * 4 + 2]);
            __m512h x3 = _mm512_set1_ph(xh[(size_t)(i + s) * 4 + 3]);
            __m512h mx = _mm512_setzero_ph();
            for (int k = 0; k < GGML_PQ_K; k += 32) {
                __m512h v = _mm512_mul_ph(x0, _mm512_loadu_ph(cs + k));
                v = _mm512_fmadd_ph(x1, _mm512_loadu_ph(cs + GGML_PQ_K + k), v);
                v = _mm512_fmadd_ph(x2, _mm512_loadu_ph(cs + 2 * GGML_PQ_K + k), v);
                v = _mm512_fmadd_ph(x3, _mm512_loadu_ph(cs + 3 * GGML_PQ_K + k), v);
                _mm512_storeu_ph(dt[s] + k, v);
                mx = _mm512_max_ph(mx, _mm512_abs_ph(v));
            }
            const float amax = _mm512_reduce_max_ph(mx);
            const float scale = amax > 0.f ? 127.f / amax : 0.f;
            inv[s] = _mm512_set1_ph((_Float16)(amax > 0.f ? amax / 127.f : 0.f));
            const __m512h sc = _mm512_set1_ph((_Float16) scale);
            for (int q = 0; q < 4; q++) {
                const __m512i lo = _mm512_cvtph_epi16(_mm512_mul_ph(_mm512_loadu_ph(dt[s] + q * 64), sc));
                const __m512i hi = _mm512_cvtph_epi16(_mm512_mul_ph(_mm512_loadu_ph(dt[s] + q * 64 + 32), sc));
                lut[s][q] = _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm512_cvtepi16_epi8(lo)),
                    _mm512_cvtepi16_epi8(hi), 1);
            }
        }
        for (int j = 0; j < dout; j += 64) {
            const int n = std::min(64, dout - j);
            if (n < 64) {
                for (int k = j; k < dout; k++) {
                    float v = (float) yl[k];
                    for (int s = 0; s < 4; s++) v += (float) dt[s][idx[(size_t)(i + s) * dout + k]];
                    yl[k] = (_Float16) v;
                }
                break;
            }
            __m512h acc0 = _mm512_loadu_ph(yl + j);
            __m512h acc1 = _mm512_loadu_ph(yl + j + 32);
            for (int s = 0; s < 4; s++) {
                const __m512i iv = _mm512_loadu_si512((const void *)(idx + (size_t)(i + s) * dout + j));
                const __m512i q0 = _mm512_permutex2var_epi8(lut[s][0], iv, lut[s][1]);
                const __m512i q1 = _mm512_permutex2var_epi8(lut[s][2], _mm512_xor_si512(iv, sign), lut[s][3]);
                const __m512i q = _mm512_mask_blend_epi8(_mm512_test_epi8_mask(iv, sign), q0, q1);
                acc0 = _mm512_fmadd_ph(_mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(q))), inv[s], acc0);
                acc1 = _mm512_fmadd_ph(_mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(q, 1))), inv[s], acc1);
            }
            _mm512_storeu_ph(yl + j, acc0);
            _mm512_storeu_ph(yl + j + 32, acc1);
        }
    }
    if (i < i1) pq_s1_phase1_impl<4>(t, xh, i, i1, yl);
}

static void pq_s1_phase1(const PQTensor & t, const _Float16 * xh,
                         int i0, int i1, _Float16 * yl) {
    switch (t.ds) {
    case 1: pq_s1_phase1_impl<1>(t, xh, i0, i1, yl); break;
    case 2: pq_s1_phase1_impl<2>(t, xh, i0, i1, yl); break;
    case 3: pq_s1_phase1_impl<3>(t, xh, i0, i1, yl); break;
    case 4:
        if (getenv("GGML_PQ_DS4_GENERIC")) pq_s1_phase1_impl<4>(t, xh, i0, i1, yl);
        else pq_s1_phase1_ds4_block(t, xh, i0, i1, yl);
        break;
    case 5: pq_s1_phase1_impl<5>(t, xh, i0, i1, yl); break;
    case 6: pq_s1_phase1_impl<6>(t, xh, i0, i1, yl); break;
    case 7: pq_s1_phase1_impl<7>(t, xh, i0, i1, yl); break;
    case 8: pq_s1_phase1_impl<8>(t, xh, i0, i1, yl); break;
    default: break;
    }
}

// phase 2: reduce every thread's partial row for `t` into dst[j0..j1)
static void pq_s1_phase2(const PQTensor & t, float * dst, int j0, int j1,
                         int nth, const _Float16 * base) {
    const size_t row = (size_t)t.n_out;
    int j = j0;
    for (; j + 16 <= j1; j += 16) {
        __m512 sum = _mm512_setzero_ps();
        for (int p = 0; p < nth; p++) {
            const _Float16 * src = base + (size_t) p * row + j;
            sum = _mm512_add_ps(sum, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) src)));
        }
        if (t.scaled) {
            sum = _mm512_mul_ps(sum, _mm512_loadu_ps(t.row_scale.data() + j));
        }
        _mm512_storeu_ps(dst + j, sum);
    }
    for (; j < j1; j++) {
        float sum = 0.f;
        for (int p = 0; p < nth; p++)
            sum += (float)base[(size_t)p * row + j];
        dst[j] = t.scaled ? sum * t.row_scale[(size_t) j] : sum;
    }
}

// pending same-input S1 fusion group (per-thread slot; all threads run the
// identical node sequence, so their slots stay in lockstep without sharing)
struct S1Group {
    bool active = false;
    const void * key = nullptr;   // shared input tensor (fusion identity)
    int n = 0;
    const PQTensor * t[4];
    float * dst[4];
    const _Float16 * xh[4];       // fp16 copy of the input, captured at
                                  // append time (the source buffer may be
                                  // recycled by the allocator before flush)
};
S1Group g_s1_group[kMaxThreads];

// append a member and snapshot its input row in fp16 (must run while the
// source buffer is still alive, i.e. inside the member's own node)
static void pq_s1_group_add(S1Group & g, const PQTensor & t, const void * key,
                            const void * x, int x_type, float * dst) {
    const int m = g.n++;
    g.t[m] = &t;
    g.dst[m] = dst;
    g.key = key;
    g.active = true;
    // per-thread scratch, one slot per member
    thread_local std::vector<_Float16> mem[4];
    if ((int)mem[m].size() < (int)t.n_in) mem[m].resize(t.n_in);
    if (x_type == 1 /* GGML_TYPE_F16 */) {
        const _Float16 * xf = (const _Float16 *)x;
        for (int j = 0; j < (int)t.n_in; j++) mem[m][j] = xf[j];
    } else {
        const float * xf = (const float *)x;
        for (int j = 0; j < (int)t.n_in; j++) mem[m][j] = (_Float16)xf[j];
    }
    g.xh[m] = mem[m].data();
}

// compute all members of the group in one phase1 / barrier / phase2 pass.
// sync=true (deferred flush running inside a CONSUMER node's boundary):
// a trailing barrier is required so no thread starts the consumer before
// every thread finished writing its phase-2 output slices.
static void pq_s1_flush(S1Group & g, int ith, int nth, void * threadpool,
                        bool sync) {
    if (!g.active) return;
    g.active = false;
    const int n = g.n;

    const uint64_t tq0 = (ith == 0) ? ggml_time_us() : 0;

    size_t base[4];
    size_t need = 0;
    int Dtot = 0;
    for (int m = 0; m < n; m++) {
        base[m]  = need;
        need    += (size_t)nth * g.t[m]->n_out;
        Dtot    += (int)g.t[m]->n_out;
    }
    if (g_s1_partials.size() < need) {
        // grow defensively (pool is pre-sized at registration)
        std::lock_guard<std::mutex> lk(g_s1_mu);
        if (g_s1_partials.size() < need)
            g_s1_partials.resize(need);
    }

    // phase 1: each member's subspaces are partitioned across ALL threads
    // (identical accumulation order to the unfused kernel — fp16 partial
    // precision is sensitive to the per-thread chain length)
    int m0 = 0;
    for (int m = 0; m < n; m++) {
        const PQTensor & t = *g.t[m];
        const int M = (int)(t.n_in / t.ds);
        const int a0 = (int)((int64_t)M * ith / nth);
        const int a1 = (int)((int64_t)M * (ith + 1) / nth);
        _Float16 * my = g_s1_partials.data() + base[m] + (size_t)ith * t.n_out;
        // zero unconditionally: phase 2 reads every thread's row
        std::fill(my, my + t.n_out, (_Float16)0);
        if (a1 > a0) {
            pq_s1_phase1(t, g.xh[m], a0, a1, my);
        }
        m0 += M;
    }

    ggml_barrier((struct ggml_threadpool *)threadpool);

    // phase 2: global output partition across members
    const int j0 = (int)((int64_t)Dtot * ith / nth);
    const int j1 = (int)((int64_t)Dtot * (ith + 1) / nth);
    int d0 = 0;
    for (int m = 0; m < n; m++) {
        const PQTensor & t = *g.t[m];
        const int b0 = std::max(j0, d0), b1 = std::min(j1, d0 + (int)t.n_out);
        if (b1 > b0)
            pq_s1_phase2(t, g.dst[m], b0 - d0, b1 - d0, nth,
                         g_s1_partials.data() + base[m]);
        d0 += (int)t.n_out;
    }
    if (sync) {
        ggml_barrier((struct ggml_threadpool *)threadpool);
    }
    g.n = 0;
    g.active = false;

    if (ith == 0) {
        const uint64_t dt = ggml_time_us() - tq0;
        for (int m = 0; m < n; m++)
            ggml_pq_timing_note_mulmat(g.t[m]->name.c_str(), dt);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// C API implementations (AVX-512 path)
// ---------------------------------------------------------------------------
void ggml_pq_reset() {
    std::lock_guard<std::mutex> lk(g_s1_mu);
    g_pq.clear();
    g_pq_enabled.store(false);
}

void ggml_pq_set_enabled(bool v) { g_pq_enabled.store(v); }
bool ggml_pq_enabled(void)      { return g_pq_enabled.load(); }

bool ggml_pq_register(const char * name, int mode, int ds,
                      const float * cb_f32,
                      const int8_t * cb8, const float * inv,
                      const uint8_t * idx,
                      int64_t n_in, int64_t n_out) {
    if (!name || ds < 1 || ds > GGML_PQ_MAX_DS) return false;
    PQTensor t;
    t.name = name;
    t.mode = mode;
    t.ds = ds;
    t.n_in = n_in;
    t.n_out = n_out;
    if (mode == 0) {
        if (!cb_f32 || !idx) return false;
        {
            // build fp16 SoA codebook in a temp vector, then land it on huge pages
            std::vector<_Float16> tmp((size_t)(n_in / ds) * ds * GGML_PQ_K);
            for (int64_t i = 0; i < n_in / ds; i++)
                for (int d = 0; d < ds; d++)
                    for (int k = 0; k < GGML_PQ_K; k++)
                        tmp[((size_t)(i * ds + d) * GGML_PQ_K) + k] = (_Float16)cb_f32[((size_t)i * GGML_PQ_K + k) * ds + d];
            t.cbh.assign(tmp.data(), tmp.size());
        }
        t.idx.assign(idx, (size_t)(n_in / ds) * n_out);
    } else {
        if (!cb8 || !inv || !idx) return false;
        t.cb8.assign(cb8, (size_t)(n_out / ds) * ds * GGML_PQ_K);
        t.inv.resize((size_t)(n_out / ds) * ds);
        for (size_t s = 0; s < t.inv.size(); s++) t.inv[s] = inv[s] * 128.0f;
        t.cb8u.assign(t.cb8.data(), t.cb8.size());
        for (size_t k = 0; k < t.cb8u.size(); k++)
            t.cb8u.data()[k] = (int8_t)((int)t.cb8.data()[k] + 128);
        t.idx.assign(idx, (size_t)(n_out / ds) * n_in);
    }
    g_pq[name] = std::move(t);

    if (mode == 0) {
        // pre-size for a worst-case fused group of 4 same-shape tensors
        std::lock_guard<std::mutex> lk(g_s1_mu);
        const size_t want = 4 * kMaxThreads * (size_t)n_out;
        if (g_s1_partials.size() < want)
            g_s1_partials.assign(want, (_Float16)0);
    }
    return true;
}

bool ggml_pq_register_raw(const char * name, int mode, int ds,
                          const void * cb,
                          const float * inv,
                          const uint8_t * idx,
                          int64_t n_in, int64_t n_out) {
    if (!name || ds < 1 || ds > GGML_PQ_MAX_DS || !cb || !idx) return false;
    PQTensor t;
    t.name = name;
    t.mode = mode;
    t.ds = ds;
    t.n_in = n_in;
    t.n_out = n_out;
    if (mode == 0) {
        // fp16 SoA [(i*ds+d)*K+k] — direct copy
        const _Float16 * cbh = (const _Float16 *)cb;
        t.cbh.assign(cbh, (size_t)(n_in / ds) * ds * GGML_PQ_K);
        t.idx.assign(idx, (size_t)(n_in / ds) * n_out);
    } else {
        // int8 codebook + float scales (scaled by 128 here)
        t.cb8.assign((const int8_t *)cb, (size_t)(n_out / ds) * ds * GGML_PQ_K);
        t.inv.resize((size_t)(n_out / ds) * ds);
        for (size_t s = 0; s < t.inv.size(); s++) t.inv[s] = inv[s] * 128.0f;
        t.cb8u.assign(t.cb8.data(), t.cb8.size());
        for (size_t k = 0; k < t.cb8u.size(); k++)
            t.cb8u.data()[k] = (int8_t)((int)t.cb8.data()[k] + 128);
        t.idx.assign(idx, (size_t)(n_out / ds) * n_in);
    }
    g_pq[name] = std::move(t);

    if (mode == 0) {
        // pre-size for a worst-case fused group of 4 same-shape tensors
        std::lock_guard<std::mutex> lk(g_s1_mu);
        const size_t want = 4 * kMaxThreads * (size_t)n_out;
        if (g_s1_partials.size() < want)
            g_s1_partials.assign(want, (_Float16)0);
    }
    return true;
}

bool ggml_pq_register_raw_scaled(const char * name, int ds,
                                 const void * cb, const uint8_t * idx,
                                 const float * row_scale,
                                 int64_t n_in, int64_t n_out) {
    if (!name || ds < 1 || ds > GGML_PQ_MAX_DS || !cb || !idx || !row_scale || n_in % ds != 0) return false;
    PQTensor t;
    t.name = name; t.mode = 0; t.ds = ds; t.n_in = n_in; t.n_out = n_out;
    t.cbh.assign((const _Float16 *) cb, (size_t) n_in * GGML_PQ_K);
    t.idx.assign(idx, (size_t) (n_in / ds) * n_out);
    t.row_scale.assign(row_scale, row_scale + n_out);
    t.scaled = true;
    g_pq[name] = std::move(t);
    std::lock_guard<std::mutex> lk(g_s1_mu);
    const size_t want = 4 * kMaxThreads * (size_t) n_out;
    if (g_s1_partials.size() < want) g_s1_partials.assign(want, (_Float16) 0);
    return true;
}

// fp16 x scaled by 1/128 (S2 kernel input; keeps the fp16 accumulator in range),
// plus the int8-quantized x for the VNNI kernel:  xq = round(x * xs),
// sx = sum of xq over the VNNI-covered range (din rounded down to 64).
static S2XBufs pq_s2_prescale_x(const PQTensor & t, const void * x, int x_type) {
    thread_local std::vector<_Float16> xh;
    thread_local std::vector<int8_t> xq;
    const int64_t din = t.n_in;
    if ((int)xh.size() < (int)din) { xh.resize(din); xq.resize(din); }
    _Float16 * xhp = xh.data();
    int8_t * xqp = xq.data();
    const __m512 vscale = _mm512_set1_ps(1.0f / 128.0f);
    __m512 vmx = _mm512_setzero_ps();
    int64_t j = 0;
    if (x_type == 1 /* GGML_TYPE_F16 */) {
        const _Float16 * xf = (const _Float16 *)x;
        for (; j + 16 <= din; j += 16) {
            // legacy intrinsics only: __m256i load -> cvtph_ps -> __m512
            const __m512 v = _mm512_mul_ps(_mm512_cvtph_ps(
                _mm256_loadu_si256((const __m256i *)(xf + j))), vscale);
            vmx = _mm512_max_ps(vmx, _mm512_abs_ps(v));
            // fp32 -> fp16 pack (legacy cvtps_ph -> __m256i), store 16 xh
            _mm256_storeu_si256((__m256i *)(xhp + j),
                _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
        }
    } else {
        const float * xf = (const float *)x;
        for (; j + 16 <= din; j += 16) {
            const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(xf + j), vscale);
            vmx = _mm512_max_ps(vmx, _mm512_abs_ps(v));
            _mm256_storeu_si256((__m256i *)(xhp + j),
                _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
        }
    }
    float mx = _mm512_reduce_max_ps(vmx);
    for (; j < din; j++) {
        float v;
        if (x_type == 1) v = (float)((const _Float16 *)x)[j] / 128.0f;
        else v = ((const float *)x)[j] / 128.0f;
        xhp[j] = (_Float16)v;
        mx = fmaxf(mx, fabsf(v));
    }
    // xh is fp16 storage but we keep an fp32 shadow in xh32 for the pack step:
    // (re-derive xq directly from xh bits via cvtph_ps is fine)
    const float xs = mx > 0 ? 127.0f / mx : 1.0f;
    int64_t sx = 0;
    {
        const __m512 vs = _mm512_set1_ps(xs);
        const __m512 v127 = _mm512_set1_ps(127.0f);
        const __m512 v127n = _mm512_set1_ps(-127.0f);
        __m512i vsx = _mm512_setzero_si512();
        int64_t k = 0;
        for (; k + 16 <= din; k += 16) {
            const __m512 v = _mm512_mul_ps(_mm512_cvtph_ps(
                _mm256_loadu_si256((const __m256i *)(xhp + k))), vs);
            const __m512 q = _mm512_max_ps(_mm512_min_ps(v, v127), v127n);
            const __m512i qi = _mm512_cvtps_epi32(q);      // round-to-nearest
            vsx = _mm512_add_epi32(vsx, qi);
            const __m256i p16 = _mm512_cvtepi32_epi16(qi);
            _mm_storeu_si128((__m128i *)(xqp + k), _mm256_cvtepi16_epi8(p16));
        }
        sx = _mm512_reduce_add_epi32(vsx);
        for (; k < din; k++) {
            const int8_t q = (int8_t) std::lround((float)xhp[k] * xs);
            xqp[k] = q;
            sx += q;
        }
    }
    return { xh.data(), xq.data(), xs, sx };
}

bool ggml_pq_mul_mat_vec(const char * name, const void * x, int x_type,
                         float * dst, int64_t n_in, int64_t n_out,
                         int ith, int nth, void * threadpool) {
    if (!g_pq_enabled.load(std::memory_order_relaxed)) return false;
    auto it = g_pq.find(name);
    if (it == g_pq.end()) return false;
    const PQTensor & t = it->second;
    if (t.n_in != n_in || t.n_out != n_out) return false;

    if (t.mode == 1) {
        pq_s2_gemv(t, pq_s2_prescale_x(t, x, x_type), dst, ith, nth);
    } else {
        S1Group g;
        pq_s1_group_add(g, t, nullptr, x, x_type, dst);
        pq_s1_flush(g, ith, nth, threadpool, false);
    }
    return true;
}

// decode-shape MUL_MAT conditions mirrored from the ggml_compute_forward hook
static bool pq_node_is_decode_mul_mat(const struct ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT) return false;
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1) return false;
    return node->type == GGML_TYPE_F32 &&
           src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           src1->ne[0] == src0->ne[0] &&
           src1->nb[0] == ggml_type_size(src1->type) &&
           node->nb[1] == node->ne[0] * sizeof(float) &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
}

int ggml_pq_mul_mat_fused(const char * name, const void * key, const void * x,
                          int x_type, float * dst, int64_t n_in, int64_t n_out,
                          int ith, int nth, void * threadpool) {
    static const bool g_dis_s1 = getenv("GGML_PQ_DISABLE_S1") != nullptr;
    static const bool g_dis_s2 = getenv("GGML_PQ_DISABLE_S2") != nullptr;
    if (!g_pq_enabled.load(std::memory_order_relaxed) || g_pq.empty())
        return GGML_PQ_MM_NONE;
    auto it = g_pq.find(name);
    if (it == g_pq.end()) return GGML_PQ_MM_NONE;
    PQTensor & t = it->second;
    if (t.n_in != n_in || t.n_out != n_out) return GGML_PQ_MM_NONE;
    if (t.mode == 0 && g_dis_s1) return GGML_PQ_MM_NONE;
    if (t.mode == 1 && g_dis_s2) return GGML_PQ_MM_NONE;
    static const bool g_dis_attn = getenv("GGML_PQ_DISABLE_ATTN") != nullptr;
    static const bool g_dis_ffn  = getenv("GGML_PQ_DISABLE_FFN") != nullptr;
    if (g_dis_attn && strstr(name, ".attn_")) return GGML_PQ_MM_NONE;
    if (g_dis_ffn && strstr(name, ".ffn_")) return GGML_PQ_MM_NONE;

    if (t.mode == 2) {
        // graph-level fused node: the members' own nodes no longer exist, so
        // the group always runs as one immediate pass (the consumer is in a
        // later node; the node-end barrier suffices, no trailing sync needed)
        if (ith >= kMaxThreads) return GGML_PQ_MM_NONE;
        S1Group & g = g_s1_group[ith];
        pq_s1_flush(g, ith, nth, threadpool, true);
        for (int m = 0; m < t.n_seg; m++) {
            pq_s1_group_add(g, *t.seg_t[m], key, x, x_type, dst + t.seg_off[m]);
        }
        pq_s1_flush(g, ith, nth, threadpool, false);
        return GGML_PQ_MM_DONE;
    }

    if (ith >= kMaxThreads) {
        // oversized pool: no grouping, compute immediately
        if (t.mode == 1) {
            pq_s2_gemv(t, pq_s2_prescale_x(t, x, x_type), dst, ith, nth);
        } else {
            S1Group g;
            pq_s1_group_add(g, t, key, x, x_type, dst);
            pq_s1_flush(g, ith, nth, threadpool, false);
        }
        return GGML_PQ_MM_DONE;
    }

    S1Group & g = g_s1_group[ith];
    if (t.mode == 1) {
        // S2 consumes the pending group's outputs: flush first
        pq_s1_flush(g, ith, nth, threadpool, true);
        pq_s2_gemv(t, pq_s2_prescale_x(t, x, x_type), dst, ith, nth);
        return GGML_PQ_MM_DONE;
    }

    static const bool g_no_fuse = getenv("GGML_PQ_NO_FUSE") != nullptr;
    if (g_no_fuse) {
        pq_s1_flush(g, ith, nth, threadpool, false);
        pq_s1_group_add(g, t, key, x, x_type, dst);
        pq_s1_flush(g, ith, nth, threadpool, false);
        return GGML_PQ_MM_DONE;
    }
    if (g.active && g.key == key && g.n < 4) {
        pq_s1_group_add(g, t, key, x, x_type, dst);
        return GGML_PQ_MM_DEFERRED;
    }
    pq_s1_flush(g, ith, nth, threadpool, true);
    pq_s1_group_add(g, t, key, x, x_type, dst);
    return GGML_PQ_MM_DEFERRED;
}

void ggml_pq_node_boundary(const void * node_ptr, int ith, int nth, void * threadpool) {
    if (!g_pq_enabled.load(std::memory_order_relaxed) || g_pq.empty()) return;
    if (ith >= kMaxThreads) return;
    // decode-shaped MUL_MAT on a registered tensor manages grouping itself
    const struct ggml_tensor * node = (const struct ggml_tensor *) node_ptr;
    if (node && pq_node_is_decode_mul_mat(node) && g_pq.count(node->src[0]->name))
        return;
    pq_s1_flush(g_s1_group[ith], ith, nth, threadpool, true);
}

void ggml_pq_graph_end(int ith, int nth, void * threadpool) {
    if (!g_pq_enabled.load(std::memory_order_relaxed)) return;
    if (ith >= kMaxThreads) return;
    pq_s1_flush(g_s1_group[ith], ith, nth, threadpool, true);
}

bool ggml_pq_register_group(const char * name, int64_t n_in, int64_t n_out_total,
                            const char * const * members, int n_members,
                            const int64_t * row_offs) {
    if (!name || n_members < 1 || n_members > 4) return false;
    PQTensor g;
    g.name = name;
    g.mode = 2;
    g.n_in = n_in;
    g.n_out = n_out_total;
    for (int m = 0; m < n_members; m++) {
        auto it = g_pq.find(members[m]);
        if (it == g_pq.end() || it->second.mode != 0) return false;
        const PQTensor & t = it->second;
        if (t.n_in != n_in) return false;
        if (g.n_seg > 0 && t.ds != g.ds) return false;
        g.ds = t.ds;
        g.seg_t[g.n_seg]  = &t;
        g.seg_off[g.n_seg] = row_offs[m];
        g.n_seg++;
    }
    g_pq[name] = std::move(g);
    return true;
}

void ggml_pq_timing_note_mulmat(const char * src_name, uint64_t busy_us) {
    std::lock_guard<std::mutex> lk(g_mulmat_mu);
    auto & a = g_mulmat_acc[src_name];
    a.busy_us += busy_us;
    a.count++;
}

void ggml_pq_timing_dump_mulmat(void) {
    std::vector<std::pair<std::string, MulmatAcc>> top;
    {
        std::lock_guard<std::mutex> lk(g_mulmat_mu);
        top.assign(g_mulmat_acc.begin(), g_mulmat_acc.end());
        g_mulmat_acc.clear();
    }
    std::sort(top.begin(), top.end(),
              [](const auto & a, const auto & b){ return a.second.busy_us > b.second.busy_us; });
    fprintf(stderr, "  top MUL_MAT by busy:\n");
    for (size_t i = 0; i < top.size() && i < 8; i++)
        fprintf(stderr, "    %-36s n=%6llu  busy=%8.2f ms (avg %.0f us)\n",
                top[i].first.c_str(), (unsigned long long)top[i].second.count,
                top[i].second.busy_us / 1000.0,
                top[i].second.busy_us / (double)top[i].second.count);
}

#else // !AVX512FP16/VBMI

void ggml_pq_reset(void) {}
void ggml_pq_set_enabled(bool) {}
bool ggml_pq_enabled(void) { return false; }
bool ggml_pq_register(const char *, int, int, const float *, const int8_t *,
                      const float *, const uint8_t *, int64_t, int64_t) { return false; }
bool ggml_pq_register_raw(const char *, int, int, const void *, const float *,
                          const uint8_t *, int64_t, int64_t) { return false; }
bool ggml_pq_register_raw_scaled(const char *, int, const void *, const uint8_t *,
                                 const float *, int64_t, int64_t) { return false; }
bool ggml_pq_mul_mat_vec(const char *, const void *, int, float *,
                         int64_t, int64_t, int, int, void *) { return false; }
int ggml_pq_mul_mat_fused(const char *, const void *, const void *, int, float *,
                          int64_t, int64_t, int, int, void *) { return GGML_PQ_MM_NONE; }
void ggml_pq_node_boundary(const void *, int, int, void *) {}
void ggml_pq_graph_end(int, int, void *) {}
bool ggml_pq_register_group(const char *, int64_t, int64_t,
                            const char * const *, int, const int64_t *) { return false; }
void ggml_pq_timing_note_mulmat(const char *, uint64_t) {}
void ggml_pq_timing_dump_mulmat(void) {}

#endif // __AVX512FP16__
