// PQ decode-time GEMV for ARM64 (Huawei Kunpeng / aarch64).
//
// Mirrors the public ggml_pq_* API in pq-gemv.cpp (AVX-512 x86 path).
// Hot path uses NEON 256-entry byte LUT (+ optional SDOT). SVE is reserved
// for a later width-specialized kernel; ggml's own SVE paths remain unchanged.
#include "ggml-pq.h"

#if defined(__aarch64__)

#include <arm_neon.h>
#include <sys/mman.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ggml-cpu-impl.h"
#include "ggml.h"

namespace {

using pq_f16 = ggml_fp16_t;

static inline float pq_to_f32(pq_f16 v) { return ggml_fp16_to_fp32(v); }
static inline pq_f16 pq_from_f32(float v) { return ggml_fp32_to_fp16(v); }

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
        void * raw = mmap(nullptr, bytes + HP, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) {
            fprintf(stderr, "pq-arm: mmap failed\n");
            exit(1);
        }
        ptr = (T *)(((uintptr_t) raw + HP - 1) & ~(uintptr_t)(HP - 1));
        madvise(ptr, bytes, MADV_HUGEPAGE);
        memcpy(ptr, src, n * sizeof(T));
    }
    huge_vec() = default;
    huge_vec(huge_vec && o) noexcept : ptr(o.ptr), n(o.n) { o.ptr = nullptr; o.n = 0; }
    huge_vec & operator=(huge_vec && o) noexcept {
        if (this != &o) {
            free_mem();
            ptr = o.ptr;
            n = o.n;
            o.ptr = nullptr;
            o.n = 0;
        }
        return *this;
    }
    T * data() { return ptr; }
    const T * data() const { return ptr; }
    size_t size() const { return n; }
    ~huge_vec() { free_mem(); }
};

struct PQTensor {
    std::string name;
    int      mode = 0;
    int      ds = 2;
    int64_t  n_in = 0, n_out = 0;
    int             n_seg = 0;
    const PQTensor * seg_t[4];
    int64_t          seg_off[4];
    huge_vec<pq_f16> cbh;
    huge_vec<int8_t>   cb8;
    huge_vec<int8_t>   cb8u;
    std::vector<float> inv;
    huge_vec<uint8_t>  idx;
    std::vector<float> row_scale;
    bool scaled = false;
};

std::unordered_map<std::string, PQTensor> g_pq;
std::atomic<bool> g_pq_enabled{false};

constexpr int kMaxThreads = 256;

std::vector<pq_f16> g_s1_partials;
std::mutex          g_s1_mu;

struct MulmatAcc { uint64_t busy_us = 0, count = 0; };
static std::map<std::string, MulmatAcc> g_mulmat_acc;
static std::mutex g_mulmat_mu;

// ---------------------------------------------------------------------------
// NEON helpers: true 256-entry byte LUT (vqtbl1 alone only covers 16 entries).
// Four vqtbl4q tables cover bytes [0,64), [64,128), [128,192), [192,256).
// Out-of-range lanes are zeroed by the table instruction, so OR merges safely.
// ---------------------------------------------------------------------------
static inline uint8x16_t pq_neon_tbl256(const uint8_t * table, uint8x16_t idx) {
    const uint8x16x4_t t0 = vld1q_u8_x4(table + 0);
    const uint8x16x4_t t1 = vld1q_u8_x4(table + 64);
    const uint8x16x4_t t2 = vld1q_u8_x4(table + 128);
    const uint8x16x4_t t3 = vld1q_u8_x4(table + 192);
    const uint8x16_t r0 = vqtbl4q_u8(t0, idx);
    const uint8x16_t r1 = vqtbl4q_u8(t1, vsubq_u8(idx, vdupq_n_u8(64)));
    const uint8x16_t r2 = vqtbl4q_u8(t2, vsubq_u8(idx, vdupq_n_u8(128)));
    const uint8x16_t r3 = vqtbl4q_u8(t3, vsubq_u8(idx, vdupq_n_u8(192)));
    return vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3));
}

static inline int8x16_t pq_neon_tbl256_s8(const int8_t * table, uint8x16_t idx) {
    return vreinterpretq_s8_u8(pq_neon_tbl256((const uint8_t *) table, idx));
}

#if defined(__ARM_FEATURE_DOTPROD)
static inline int32x4_t pq_neon_sdot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vdotq_s32(acc, a, b);
}
#else
// Portable fallback: signed int8 x signed int8 -> int32 accumulate (16 lanes).
static inline int32x4_t pq_neon_sdot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t al = vmovl_s8(vget_low_s8(a));
    const int16x8_t ah = vmovl_high_s8(a);
    const int16x8_t bl = vmovl_s8(vget_low_s8(b));
    const int16x8_t bh = vmovl_high_s8(b);
    acc = vaddq_s32(acc, vmull_s16(vget_low_s16(al), vget_low_s16(bl)));
    acc = vaddq_s32(acc, vmull_high_s16(al, bl));
    acc = vaddq_s32(acc, vmull_s16(vget_low_s16(ah), vget_low_s16(bh)));
    acc = vaddq_s32(acc, vmull_high_s16(ah, bh));
    return acc;
}
#endif

static void pq_build_dt_scalar(int ds, const pq_f16 * xh, const pq_f16 * cbh,
                               int subspace, float * dt) {
    for (int k = 0; k < GGML_PQ_K; k++) {
        float s = 0.f;
        for (int d = 0; d < ds; d++) {
            s += pq_to_f32(xh[(size_t) subspace * ds + d]) *
                 pq_to_f32(cbh[((size_t)(subspace * ds + d) * GGML_PQ_K) + k]);
        }
        dt[k] = s;
    }
}

static void pq_s1_accumulate_subspace(const PQTensor & t, const pq_f16 * xh,
                                      int subspace, pq_f16 * yl) {
    const int ds = t.ds;
    const int dout = (int) t.n_out;
    const pq_f16 * cbh = t.cbh.data();
    const uint8_t * idx = t.idx.data();

    float dt[GGML_PQ_K];
    pq_build_dt_scalar(ds, xh, cbh, subspace, dt);

    float amax = 0.f;
    for (int k = 0; k < GGML_PQ_K; k++) {
        amax = fmaxf(amax, fabsf(dt[k]));
    }
    const float inv = amax > 0.f ? amax / 127.0f : 0.f;
    int8_t dt8[GGML_PQ_K];
    if (amax > 0.f) {
        const float sc = 127.0f / amax;
        for (int k = 0; k < GGML_PQ_K; k++) {
            dt8[k] = (int8_t) std::lround(dt[k] * sc);
        }
    } else {
        memset(dt8, 0, sizeof(dt8));
    }

    const uint8_t * ii = idx + (size_t) subspace * dout;
    int j = 0;
    for (; j + 15 < dout; j += 16) {
        const uint8x16_t iv = vld1q_u8(ii + j);
        const int8x16_t qv = pq_neon_tbl256_s8(dt8, iv);
        const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
        const int16x8_t qhi = vmovl_high_s8(qv);
        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo)));
        float32x4_t f1 = vcvtq_f32_s32(vmovl_high_s16(qlo));
        float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi)));
        float32x4_t f3 = vcvtq_f32_s32(vmovl_high_s16(qhi));
        f0 = vmulq_n_f32(f0, inv);
        f1 = vmulq_n_f32(f1, inv);
        f2 = vmulq_n_f32(f2, inv);
        f3 = vmulq_n_f32(f3, inv);
        float ytmp[16];
        for (int t = 0; t < 16; t++) {
            ytmp[t] = pq_to_f32(yl[j + t]);
        }
        float32x4_t y0 = vld1q_f32(ytmp + 0);
        float32x4_t y1 = vld1q_f32(ytmp + 4);
        float32x4_t y2 = vld1q_f32(ytmp + 8);
        float32x4_t y3 = vld1q_f32(ytmp + 12);
        y0 = vaddq_f32(y0, f0);
        y1 = vaddq_f32(y1, f1);
        y2 = vaddq_f32(y2, f2);
        y3 = vaddq_f32(y3, f3);
        vst1q_f32(ytmp + 0, y0);
        vst1q_f32(ytmp + 4, y1);
        vst1q_f32(ytmp + 8, y2);
        vst1q_f32(ytmp + 12, y3);
        for (int t = 0; t < 16; t++) {
            yl[j + t] = pq_from_f32(ytmp[t]);
        }
    }
    for (; j < dout; j++) {
        const float acc = pq_to_f32(yl[j]) + (float) dt8[ii[j]] * inv;
        yl[j] = pq_from_f32(acc);
    }
}

static void pq_s1_phase1(const PQTensor & t, const pq_f16 * xh,
                         int i0, int i1, pq_f16 * yl) {
    for (int i = i0; i < i1; i++) {
        pq_s1_accumulate_subspace(t, xh, i, yl);
    }
}

static void pq_s1_phase2(const PQTensor & t, float * dst, int j0, int j1,
                         int nth, const pq_f16 * base) {
    const size_t row = (size_t) t.n_out;
    for (int j = j0; j < j1; j++) {
        float sum = 0.f;
        for (int p = 0; p < nth; p++) {
            sum += pq_to_f32(base[(size_t) p * row + j]);
        }
        dst[j] = t.scaled ? sum * t.row_scale[(size_t) j] : sum;
    }
}

struct S2XBufs { const pq_f16 * xh; const int8_t * xq; float xs; int64_t sx; };

static S2XBufs pq_s2_prescale_x(const PQTensor & t, const void * x, int x_type) {
    thread_local std::vector<pq_f16> xh;
    thread_local std::vector<int8_t> xq;
    const int64_t din = t.n_in;
    if ((int64_t) xh.size() < din) {
        xh.resize(din);
        xq.resize(din);
    }

    float mx = 0.f;
    for (int64_t j = 0; j < din; j++) {
        float v;
        if (x_type == 1) {
            v = pq_to_f32(((const pq_f16 *) x)[j]) / 128.0f;
        } else {
            v = ((const float *) x)[j] / 128.0f;
        }
        xh[(size_t) j] = pq_from_f32(v);
        mx = fmaxf(mx, fabsf(v));
    }

    const float xs = mx > 0.f ? 127.0f / mx : 1.0f;
    int64_t sx = 0;
    for (int64_t j = 0; j < din; j++) {
        const int8_t q = (int8_t) std::lround(pq_to_f32(xh[(size_t) j]) * xs);
        xq[(size_t) j] = q;
        sx += q;
    }
    return { xh.data(), xq.data(), xs, sx };
}

// S2 ds==2: signed codebook LUT + signed xq (equivalent to x86 unsigned LUT
// with +128 bias and dpbusd correction, but simpler on NEON/SDOT).
static void pq_s2_gemv_ds2_neon(const PQTensor & t, const S2XBufs & xb,
                                float * dst, int i0, int i1) {
    const int din = (int) t.n_in;
    const int8_t * cb8 = t.cb8.data();
    const float * inv = t.inv.data();
    const uint8_t * idx = t.idx.data();
    const pq_f16 * xh = xb.xh;
    const int8_t * xq = xb.xq;
    const float xs = xb.xs;
    (void) xb.sx; // x86 unsigned-LUT path needs sx for +128 correction; signed NEON does not

    for (int i = i0; i < i1; i++) {
        const int8_t * ce = cb8 + (size_t) i * 512;
        const int8_t * co = ce + 256;
        const uint8_t * ii = idx + (size_t) i * din;

        int32x4_t ae = vdupq_n_s32(0);
        int32x4_t ao = vdupq_n_s32(0);
        int j = 0;
        for (; j + 15 < din; j += 16) {
            const int8x16_t xv = vld1q_s8(xq + j);
            const uint8x16_t iv = vld1q_u8(ii + j);
            const int8x16_t qe = pq_neon_tbl256_s8(ce, iv);
            const int8x16_t qo = pq_neon_tbl256_s8(co, iv);
            ae = pq_neon_sdot(ae, qe, xv);
            ao = pq_neon_sdot(ao, qo, xv);
        }
        int32_t se = (int32_t) vaddvq_s32(ae);
        int32_t so = (int32_t) vaddvq_s32(ao);
        float s0 = (float) se / xs;
        float s1 = (float) so / xs;
        for (; j < din; j++) {
            s0 += pq_to_f32(xh[j]) * ce[ii[j]];
            s1 += pq_to_f32(xh[j]) * co[ii[j]];
        }
        dst[(size_t) i * 2]     = s0 * inv[(size_t) i * 2];
        dst[(size_t) i * 2 + 1] = s1 * inv[(size_t) i * 2 + 1];
    }
}

static void pq_s2_gemv_scalar(const PQTensor & t, const S2XBufs & xb,
                              float * dst, int i0, int i1) {
    const int ds = t.ds;
    const int din = (int) t.n_in;
    const int8_t * cb8 = t.cb8.data();
    const float * inv = t.inv.data();
    const uint8_t * idx = t.idx.data();
    const pq_f16 * xh = xb.xh;

    for (int i = i0; i < i1; i++) {
        const int8_t * cbi = cb8 + (size_t) i * ds * GGML_PQ_K;
        const uint8_t * ii = idx + (size_t) i * din;
        for (int s = 0; s < ds; s++) {
            float sum = 0.f;
            for (int j = 0; j < din; j++) {
                sum += pq_to_f32(xh[j]) * cbi[(size_t) s * GGML_PQ_K + ii[j]];
            }
            dst[(size_t) i * ds + s] = sum * inv[(size_t) i * ds + s];
        }
    }
}

static void pq_s2_gemv(const PQTensor & t, const S2XBufs & xb, float * dst,
                       int ith, int nth) {
    const int M = (int) (t.n_out / t.ds);
    const int i0 = (int) ((int64_t) M * ith / nth);
    const int i1 = (int) ((int64_t) M * (ith + 1) / nth);
    if (t.ds == 2) {
        pq_s2_gemv_ds2_neon(t, xb, dst, i0, i1);
        return;
    }
    pq_s2_gemv_scalar(t, xb, dst, i0, i1);
}

struct S1Group {
    bool active = false;
    const void * key = nullptr;
    int n = 0;
    const PQTensor * t[4];
    float * dst[4];
    const pq_f16 * xh[4];
};

S1Group g_s1_group[kMaxThreads];

static void pq_copy_input_row(const PQTensor & t, const void * x, int x_type,
                              pq_f16 * out) {
    if (x_type == 1) {
        const pq_f16 * xf = (const pq_f16 *) x;
        for (int j = 0; j < (int) t.n_in; j++) {
            out[j] = xf[j];
        }
    } else {
        const float * xf = (const float *) x;
        for (int j = 0; j < (int) t.n_in; j++) {
            out[j] = pq_from_f32(xf[j]);
        }
    }
}

static void pq_s1_group_add(S1Group & g, const PQTensor & t, const void * key,
                            const void * x, int x_type, float * dst) {
    const int m = g.n++;
    g.t[m] = &t;
    g.dst[m] = dst;
    g.key = key;
    g.active = true;
    thread_local std::vector<pq_f16> mem[4];
    if ((int) mem[m].size() < (int) t.n_in) {
        mem[m].resize(t.n_in);
    }
    pq_copy_input_row(t, x, x_type, mem[m].data());
    g.xh[m] = mem[m].data();
}

static void pq_s1_flush(S1Group & g, int ith, int nth, void * threadpool, bool sync) {
    if (!g.active) {
        return;
    }
    g.active = false;
    const int n = g.n;

    const uint64_t tq0 = (ith == 0) ? ggml_time_us() : 0;

    size_t base[4];
    size_t need = 0;
    int Dtot = 0;
    for (int m = 0; m < n; m++) {
        base[m] = need;
        need += (size_t) nth * g.t[m]->n_out;
        Dtot += (int) g.t[m]->n_out;
    }
    if (g_s1_partials.size() < need) {
        std::lock_guard<std::mutex> lk(g_s1_mu);
        if (g_s1_partials.size() < need) {
            g_s1_partials.resize(need);
        }
    }

    for (int m = 0; m < n; m++) {
        const PQTensor & tm = *g.t[m];
        const int M = (int) (tm.n_in / tm.ds);
        const int a0 = (int) ((int64_t) M * ith / nth);
        const int a1 = (int) ((int64_t) M * (ith + 1) / nth);
        pq_f16 * my = g_s1_partials.data() + base[m] + (size_t) ith * tm.n_out;
        std::fill(my, my + tm.n_out, pq_from_f32(0.f));
        if (a1 > a0) {
            pq_s1_phase1(tm, g.xh[m], a0, a1, my);
        }
    }

    ggml_barrier((struct ggml_threadpool *) threadpool);

    const int j0 = (int) ((int64_t) Dtot * ith / nth);
    const int j1 = (int) ((int64_t) Dtot * (ith + 1) / nth);
    int d0 = 0;
    for (int m = 0; m < n; m++) {
        const PQTensor & tm = *g.t[m];
        const int b0 = std::max(j0, d0);
        const int b1 = std::min(j1, d0 + (int) tm.n_out);
        if (b1 > b0) {
            pq_s1_phase2(tm, g.dst[m], b0 - d0, b1 - d0, nth,
                         g_s1_partials.data() + base[m]);
        }
        d0 += (int) tm.n_out;
    }
    if (sync) {
        ggml_barrier((struct ggml_threadpool *) threadpool);
    }
    g.n = 0;
    g.active = false;

    if (ith == 0) {
        const uint64_t dt = ggml_time_us() - tq0;
        for (int m = 0; m < n; m++) {
            ggml_pq_timing_note_mulmat(g.t[m]->name.c_str(), dt);
        }
    }
}

static bool pq_node_is_decode_mul_mat(const struct ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1) {
        return false;
    }
    return node->type == GGML_TYPE_F32 &&
           src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           src1->ne[0] == src0->ne[0] &&
           src1->nb[0] == ggml_type_size(src1->type) &&
           node->nb[1] == node->ne[0] * sizeof(float) &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
}

} // namespace

void ggml_pq_reset(void) {
    std::lock_guard<std::mutex> lk(g_s1_mu);
    g_pq.clear();
    g_pq_enabled.store(false);
}

void ggml_pq_set_enabled(bool v) { g_pq_enabled.store(v); }
bool ggml_pq_enabled(void) { return g_pq_enabled.load(); }

bool ggml_pq_register(const char * name, int mode, int ds,
                      const float * cb_f32, const int8_t * cb8, const float * inv,
                      const uint8_t * idx, int64_t n_in, int64_t n_out) {
    if (!name || ds < 1 || ds > GGML_PQ_MAX_DS) {
        return false;
    }
    PQTensor t;
    t.name = name;
    t.mode = mode;
    t.ds = ds;
    t.n_in = n_in;
    t.n_out = n_out;
    if (mode == 0) {
        if (!cb_f32 || !idx) {
            return false;
        }
        std::vector<pq_f16> tmp((size_t) (n_in / ds) * ds * GGML_PQ_K);
        for (int64_t i = 0; i < n_in / ds; i++) {
            for (int d = 0; d < ds; d++) {
                for (int k = 0; k < GGML_PQ_K; k++) {
                    tmp[((size_t) (i * ds + d) * GGML_PQ_K) + k] =
                        pq_from_f32(cb_f32[((size_t) i * GGML_PQ_K + k) * ds + d]);
                }
            }
        }
        t.cbh.assign(tmp.data(), tmp.size());
        t.idx.assign(idx, (size_t) (n_in / ds) * n_out);
    } else {
        if (!cb8 || !inv || !idx) {
            return false;
        }
        t.cb8.assign(cb8, (size_t) (n_out / ds) * ds * GGML_PQ_K);
        t.inv.resize((size_t) (n_out / ds) * ds);
        for (size_t s = 0; s < t.inv.size(); s++) {
            t.inv[s] = inv[s] * 128.0f;
        }
        t.cb8u.assign(t.cb8.data(), t.cb8.size());
        for (size_t k = 0; k < t.cb8u.size(); k++) {
            t.cb8u.data()[k] = (int8_t) ((int) t.cb8.data()[k] + 128);
        }
        t.idx.assign(idx, (size_t) (n_out / ds) * n_in);
    }
    g_pq[name] = std::move(t);
    if (mode == 0) {
        std::lock_guard<std::mutex> lk(g_s1_mu);
        const size_t want = 4 * kMaxThreads * (size_t) n_out;
        if (g_s1_partials.size() < want) {
            g_s1_partials.assign(want, pq_from_f32(0.f));
        }
    }
    return true;
}

bool ggml_pq_register_raw(const char * name, int mode, int ds, const void * cb,
                          const float * inv, const uint8_t * idx,
                          int64_t n_in, int64_t n_out) {
    if (!name || ds < 1 || ds > GGML_PQ_MAX_DS || !cb || !idx) {
        return false;
    }
    PQTensor t;
    t.name = name;
    t.mode = mode;
    t.ds = ds;
    t.n_in = n_in;
    t.n_out = n_out;
    if (mode == 0) {
        t.cbh.assign((const pq_f16 *) cb, (size_t) (n_in / ds) * ds * GGML_PQ_K);
        t.idx.assign(idx, (size_t) (n_in / ds) * n_out);
    } else {
        t.cb8.assign((const int8_t *) cb, (size_t) (n_out / ds) * ds * GGML_PQ_K);
        t.inv.resize((size_t) (n_out / ds) * ds);
        for (size_t s = 0; s < t.inv.size(); s++) {
            t.inv[s] = inv[s] * 128.0f;
        }
        t.cb8u.assign(t.cb8.data(), t.cb8.size());
        for (size_t k = 0; k < t.cb8u.size(); k++) {
            t.cb8u.data()[k] = (int8_t) ((int) t.cb8.data()[k] + 128);
        }
        t.idx.assign(idx, (size_t) (n_out / ds) * n_in);
    }
    g_pq[name] = std::move(t);
    if (mode == 0) {
        std::lock_guard<std::mutex> lk(g_s1_mu);
        const size_t want = 4 * kMaxThreads * (size_t) n_out;
        if (g_s1_partials.size() < want) {
            g_s1_partials.assign(want, pq_from_f32(0.f));
        }
    }
    return true;
}

bool ggml_pq_register_raw_scaled(const char * name, int ds, const void * cb,
                                 const uint8_t * idx, const float * row_scale,
                                 int64_t n_in, int64_t n_out) {
    if (!name || ds < 1 || ds > GGML_PQ_MAX_DS || !cb || !idx || !row_scale ||
        n_in % ds != 0) {
        return false;
    }
    PQTensor t;
    t.name = name;
    t.mode = 0;
    t.ds = ds;
    t.n_in = n_in;
    t.n_out = n_out;
    t.cbh.assign((const pq_f16 *) cb, (size_t) n_in * GGML_PQ_K);
    t.idx.assign(idx, (size_t) (n_in / ds) * n_out);
    t.row_scale.assign(row_scale, row_scale + n_out);
    t.scaled = true;
    g_pq[name] = std::move(t);
    std::lock_guard<std::mutex> lk(g_s1_mu);
    const size_t want = 4 * kMaxThreads * (size_t) n_out;
    if (g_s1_partials.size() < want) {
        g_s1_partials.assign(want, pq_from_f32(0.f));
    }
    return true;
}

bool ggml_pq_mul_mat_vec(const char * name, const void * x, int x_type, float * dst,
                         int64_t n_in, int64_t n_out, int ith, int nth,
                         void * threadpool) {
    if (!g_pq_enabled.load(std::memory_order_relaxed)) {
        return false;
    }
    auto it = g_pq.find(name);
    if (it == g_pq.end()) {
        return false;
    }
    const PQTensor & t = it->second;
    if (t.n_in != n_in || t.n_out != n_out) {
        return false;
    }
    if (t.mode == 1) {
        pq_s2_gemv(t, pq_s2_prescale_x(t, x, x_type), dst, ith, nth);
    } else {
        S1Group g;
        pq_s1_group_add(g, t, nullptr, x, x_type, dst);
        pq_s1_flush(g, ith, nth, threadpool, false);
    }
    return true;
}

int ggml_pq_mul_mat_fused(const char * name, const void * key, const void * x,
                          int x_type, float * dst, int64_t n_in, int64_t n_out,
                          int ith, int nth, void * threadpool) {
    static const bool g_dis_s1 = getenv("GGML_PQ_DISABLE_S1") != nullptr;
    static const bool g_dis_s2 = getenv("GGML_PQ_DISABLE_S2") != nullptr;
    if (!g_pq_enabled.load(std::memory_order_relaxed) || g_pq.empty()) {
        return GGML_PQ_MM_NONE;
    }
    auto it = g_pq.find(name);
    if (it == g_pq.end()) {
        return GGML_PQ_MM_NONE;
    }
    PQTensor & t = it->second;
    if (t.n_in != n_in || t.n_out != n_out) {
        return GGML_PQ_MM_NONE;
    }
    if (t.mode == 0 && g_dis_s1) {
        return GGML_PQ_MM_NONE;
    }
    if (t.mode == 1 && g_dis_s2) {
        return GGML_PQ_MM_NONE;
    }
    static const bool g_dis_attn = getenv("GGML_PQ_DISABLE_ATTN") != nullptr;
    static const bool g_dis_ffn  = getenv("GGML_PQ_DISABLE_FFN") != nullptr;
    if (g_dis_attn && strstr(name, ".attn_")) {
        return GGML_PQ_MM_NONE;
    }
    if (g_dis_ffn && strstr(name, ".ffn_")) {
        return GGML_PQ_MM_NONE;
    }

    if (t.mode == 2) {
        if (ith >= kMaxThreads) {
            return GGML_PQ_MM_NONE;
        }
        S1Group & g = g_s1_group[ith];
        pq_s1_flush(g, ith, nth, threadpool, true);
        for (int m = 0; m < t.n_seg; m++) {
            pq_s1_group_add(g, *t.seg_t[m], key, x, x_type, dst + t.seg_off[m]);
        }
        pq_s1_flush(g, ith, nth, threadpool, false);
        return GGML_PQ_MM_DONE;
    }

    if (ith >= kMaxThreads) {
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
    if (!g_pq_enabled.load(std::memory_order_relaxed) || g_pq.empty()) {
        return;
    }
    if (ith >= kMaxThreads) {
        return;
    }
    const struct ggml_tensor * node = (const struct ggml_tensor *) node_ptr;
    if (node && pq_node_is_decode_mul_mat(node) && g_pq.count(node->src[0]->name)) {
        return;
    }
    pq_s1_flush(g_s1_group[ith], ith, nth, threadpool, true);
}

void ggml_pq_graph_end(int ith, int nth, void * threadpool) {
    if (!g_pq_enabled.load(std::memory_order_relaxed)) {
        return;
    }
    if (ith >= kMaxThreads) {
        return;
    }
    pq_s1_flush(g_s1_group[ith], ith, nth, threadpool, true);
}

bool ggml_pq_register_group(const char * name, int64_t n_in, int64_t n_out_total,
                            const char * const * members, int n_members,
                            const int64_t * row_offs) {
    if (!name || n_members < 1 || n_members > 4) {
        return false;
    }
    PQTensor g;
    g.name = name;
    g.mode = 2;
    g.n_in = n_in;
    g.n_out = n_out_total;
    for (int m = 0; m < n_members; m++) {
        auto it = g_pq.find(members[m]);
        if (it == g_pq.end() || it->second.mode != 0) {
            return false;
        }
        const PQTensor & tm = it->second;
        if (tm.n_in != n_in) {
            return false;
        }
        if (g.n_seg > 0 && tm.ds != g.ds) {
            return false;
        }
        g.ds = tm.ds;
        g.seg_t[g.n_seg] = &tm;
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
              [](const auto & a, const auto & b) { return a.second.busy_us > b.second.busy_us; });
    fprintf(stderr, "  top MUL_MAT by busy:\n");
    for (size_t i = 0; i < top.size() && i < 8; i++) {
        fprintf(stderr, "    %-36s n=%6llu  busy=%8.2f ms (avg %.0f us)\n",
                top[i].first.c_str(), (unsigned long long) top[i].second.count,
                top[i].second.busy_us / 1000.0,
                top[i].second.busy_us / (double) top[i].second.count);
    }
}

#endif // __aarch64__
