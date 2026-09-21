// PQ decode-time GEMV for ARM64 (Huawei Kunpeng / aarch64).
//
// Mirrors the public ggml_pq_* API in pq-gemv.cpp (AVX-512 x86 path).
// S1 (EdgePQ-4c8b, ds=4): shared LUT + output-owned accumulate by default
// (set GGML_PQ_S1_PARTIALS=1 for the legacy nth partial-y reduce). NUMA: LUT
// scratch is replicated per node (disable with GGML_PQ_NUMA_LUT=0).
#include "ggml-pq.h"

#if defined(__aarch64__)

#include <arm_neon.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <sched.h>
#include <unistd.h>
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
    int      K = GGML_PQ_K; // 256 (x86/default) or 64 (Kunpeng NEON-friendly)
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

constexpr int kMaxThreads = 1024; // Kunpeng nodes can expose 640+ logical CPUs
constexpr int kMaxNumaNodes = 8;
constexpr int kS1OutTile = 256; // amortize LUT load across a large output tile

// Legacy fp16 partials path (GGML_PQ_S1_PARTIALS=1): halves phase2 traffic vs float32.
std::vector<pq_f16> g_s1_partials;
std::mutex          g_s1_mu;

// Shared-LUT scratch: per-NUMA-node replicas of dt8[M][256] + inv[M] for up to 4
// fused S1 tensors. Phase1 writes every replica; phase2 reads the local node copy.
struct S1LutScratch {
    std::vector<int8_t> dt8;
    std::vector<float>  inv;
};
static S1LutScratch g_s1_luts[kMaxNumaNodes][4];

struct MulmatAcc { uint64_t busy_us = 0, count = 0; };
static std::map<std::string, MulmatAcc> g_mulmat_acc;
static std::mutex g_mulmat_mu;

static inline void pq_barrier(void * threadpool, int nth) {
    if (nth <= 1 || threadpool == nullptr) {
        return;
    }
    ggml_barrier((struct ggml_threadpool *) threadpool);
}

static int pq_numa_n_nodes(void) {
    static const bool g_no_numa = getenv("GGML_PQ_NUMA_LUT") != nullptr &&
                                  getenv("GGML_PQ_NUMA_LUT")[0] == '0';
    if (g_no_numa) {
        return 1;
    }
    static int cached = -1;
    if (cached >= 0) {
        return cached;
    }
    int n = 0;
    for (; n < kMaxNumaNodes; n++) {
        char path[64];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", n);
        struct stat st;
        if (stat(path, &st) != 0) {
            break;
        }
    }
    cached = n > 0 ? n : 1;
    return cached;
}

static int pq_numa_node(void) {
    unsigned cpu = 0, node = 0;
#if defined(__linux__)
    if (getcpu(&cpu, &node) == 0) {
        const int n = pq_numa_n_nodes();
        if ((int) node >= 0 && (int) node < n) {
            return (int) node;
        }
    }
#else
    (void) cpu;
    (void) node;
#endif
    return 0;
}

static bool pq_s1_use_partials(void) {
    static const bool v = getenv("GGML_PQ_S1_PARTIALS") != nullptr;
    return v;
}

static void pq_s1_ensure_luts(int n_nodes, int slot, int M, int K) {
    const size_t want_dt = (size_t) M * (size_t) K;
    for (int node = 0; node < n_nodes; node++) {
        S1LutScratch & s = g_s1_luts[node][slot];
        if (s.dt8.size() < want_dt) {
            s.dt8.resize(want_dt);
            s.inv.resize((size_t) M);
            // First-touch on the allocating thread; phase1 overwrites all rows.
            std::fill(s.dt8.begin(), s.dt8.end(), (int8_t) 0);
            std::fill(s.inv.begin(), s.inv.end(), 0.f);
        } else if ((int) s.inv.size() < M) {
            s.inv.resize((size_t) M, 0.f);
        }
    }
}

// ---------------------------------------------------------------------------
// NEON 256-entry byte LUT. Four vqtbl4q cover [0,256).
// ---------------------------------------------------------------------------
struct PqLut256 {
    uint8x16x4_t t0, t1, t2, t3;
};

static inline PqLut256 pq_lut256_load(const int8_t * table) {
    const uint8_t * u = (const uint8_t *) table;
    PqLut256 L;
    L.t0 = vld1q_u8_x4(u + 0);
    L.t1 = vld1q_u8_x4(u + 64);
    L.t2 = vld1q_u8_x4(u + 128);
    L.t3 = vld1q_u8_x4(u + 192);
    return L;
}

__attribute__((always_inline))
static inline int8x16_t pq_lut256_lookup(const PqLut256 & L, uint8x16_t idx) {
    const uint8x16_t r0 = vqtbl4q_u8(L.t0, idx);
    const uint8x16_t r1 = vqtbl4q_u8(L.t1, vsubq_u8(idx, vdupq_n_u8(64)));
    const uint8x16_t r2 = vqtbl4q_u8(L.t2, vsubq_u8(idx, vdupq_n_u8(128)));
    const uint8x16_t r3 = vqtbl4q_u8(L.t3, vsubq_u8(idx, vdupq_n_u8(192)));
    return vreinterpretq_s8_u8(vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3)));
}

// K=64: one vqtbl4q covers the full codebook (indices must be in [0,63]).
struct PqLut64 {
    uint8x16x4_t t;
};

__attribute__((always_inline))
static inline PqLut64 pq_lut64_load(const int8_t * table) {
    PqLut64 L;
    L.t = vld1q_u8_x4((const uint8_t *) table);
    return L;
}

__attribute__((always_inline))
static inline int8x16_t pq_lut64_lookup(const PqLut64 & L, uint8x16_t idx) {
    return vreinterpretq_s8_u8(vqtbl4q_u8(L.t, idx));
}

static inline void pq_prefetch_t0(const void * p) { __builtin_prefetch(p, 0, 3); }
static inline void pq_prefetch_t1(const void * p) { __builtin_prefetch(p, 0, 2); }

#if defined(__ARM_FEATURE_DOTPROD)
static inline int32x4_t pq_neon_sdot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vdotq_s32(acc, a, b);
}
#else
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

static inline float32x4_t pq_load_f16x4(const pq_f16 * p) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return vcvt_f32_f16(vld1_f16((const __fp16 *) p));
#else
    float tmp[4] = {
        pq_to_f32(p[0]), pq_to_f32(p[1]), pq_to_f32(p[2]), pq_to_f32(p[3]),
    };
    return vld1q_f32(tmp);
#endif
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
__attribute__((always_inline))
static inline void pq_i8_to_f16x8x2(int8x16_t qv, float16x8_t & lo, float16x8_t & hi) {
    const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
    const int16x8_t qhi = vmovl_high_s8(qv);
    const float16x4_t a = vcvt_f16_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))));
    const float16x4_t b = vcvt_f16_f32(vcvtq_f32_s32(vmovl_high_s16(qlo)));
    const float16x4_t c = vcvt_f16_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))));
    const float16x4_t d = vcvt_f16_f32(vcvtq_f32_s32(vmovl_high_s16(qhi)));
    lo = vcombine_f16(a, b);
    hi = vcombine_f16(c, d);
}

__attribute__((always_inline))
static inline void pq_fma_lut16_f16(float16x8_t & y0, float16x8_t & y1,
                                    const PqLut256 & L, const uint8_t * ii,
                                    float16_t inv) {
    const int8x16_t qv = pq_lut256_lookup(L, vld1q_u8(ii));
    float16x8_t lo, hi;
    pq_i8_to_f16x8x2(qv, lo, hi);
    y0 = vfmaq_n_f16(y0, lo, inv);
    y1 = vfmaq_n_f16(y1, hi, inv);
}
#endif

__attribute__((always_inline))
static inline void pq_fma_lut16_f32(float32x4_t & y0, float32x4_t & y1,
                                    float32x4_t & y2, float32x4_t & y3,
                                    const PqLut256 & L, const uint8_t * ii,
                                    float inv) {
    const int8x16_t qv = pq_lut256_lookup(L, vld1q_u8(ii));
    const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
    const int16x8_t qhi = vmovl_high_s8(qv);
    y0 = vfmaq_n_f32(y0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))), inv);
    y1 = vfmaq_n_f32(y1, vcvtq_f32_s32(vmovl_high_s16(qlo)), inv);
    y2 = vfmaq_n_f32(y2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))), inv);
    y3 = vfmaq_n_f32(y3, vcvtq_f32_s32(vmovl_high_s16(qhi)), inv);
}

static void pq_build_dt_scalar(int ds, const pq_f16 * xh, const pq_f16 * cbh,
                               int subspace, int K, float * dt) {
    for (int k = 0; k < K; k++) {
        float s = 0.f;
        for (int d = 0; d < ds; d++) {
            s += pq_to_f32(xh[(size_t) subspace * ds + d]) *
                 pq_to_f32(cbh[((size_t)(subspace * ds + d) * K) + k]);
        }
        dt[k] = s;
    }
}

static void pq_quantize_dt(const float * dt, int K, int8_t * dt8, float * inv_out) {
    float32x4_t vmax = vdupq_n_f32(0.f);
    for (int k = 0; k < K; k += 4) {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(dt + k)));
    }
    const float amax = vmaxvq_f32(vmax);
    if (!(amax > 0.f)) {
        memset(dt8, 0, (size_t) K);
        *inv_out = 0.f;
        return;
    }
    *inv_out = amax / 127.0f;
    const float32x4_t vsc = vdupq_n_f32(127.0f / amax);
    for (int k = 0; k < K; k += 16) {
        const int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(dt + k + 0), vsc));
        const int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(dt + k + 4), vsc));
        const int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(dt + k + 8), vsc));
        const int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(dt + k + 12), vsc));
        const int16x8_t p01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        const int16x8_t p23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        vst1q_s8(dt8 + k, vcombine_s8(vqmovn_s16(p01), vqmovn_s16(p23)));
    }
}

// Build dt[K] + quantize; absmax tracked during the build pass. K in {64,256}.
static void pq_build_quant_ds4(const pq_f16 * xh, const pq_f16 * cbh,
                               int subspace, int K, int8_t * dt8, float * inv_out) {
    const float32x4_t vx0 = vdupq_n_f32(pq_to_f32(xh[(size_t) subspace * 4 + 0]));
    const float32x4_t vx1 = vdupq_n_f32(pq_to_f32(xh[(size_t) subspace * 4 + 1]));
    const float32x4_t vx2 = vdupq_n_f32(pq_to_f32(xh[(size_t) subspace * 4 + 2]));
    const float32x4_t vx3 = vdupq_n_f32(pq_to_f32(xh[(size_t) subspace * 4 + 3]));
    const pq_f16 * c0 = cbh + (size_t) (subspace * 4 + 0) * K;
    const pq_f16 * c1 = cbh + (size_t) (subspace * 4 + 1) * K;
    const pq_f16 * c2 = cbh + (size_t) (subspace * 4 + 2) * K;
    const pq_f16 * c3 = cbh + (size_t) (subspace * 4 + 3) * K;

    alignas(16) float dt[GGML_PQ_K];
    float32x4_t vmax = vdupq_n_f32(0.f);
    for (int k = 0; k < K; k += 8) {
        float32x4_t v0 = vmulq_f32(vx0, pq_load_f16x4(c0 + k));
        v0 = vfmaq_f32(v0, vx1, pq_load_f16x4(c1 + k));
        v0 = vfmaq_f32(v0, vx2, pq_load_f16x4(c2 + k));
        v0 = vfmaq_f32(v0, vx3, pq_load_f16x4(c3 + k));
        float32x4_t v1 = vmulq_f32(vx0, pq_load_f16x4(c0 + k + 4));
        v1 = vfmaq_f32(v1, vx1, pq_load_f16x4(c1 + k + 4));
        v1 = vfmaq_f32(v1, vx2, pq_load_f16x4(c2 + k + 4));
        v1 = vfmaq_f32(v1, vx3, pq_load_f16x4(c3 + k + 4));
        vst1q_f32(dt + k, v0);
        vst1q_f32(dt + k + 4, v1);
        vmax = vmaxq_f32(vmax, vabsq_f32(v0));
        vmax = vmaxq_f32(vmax, vabsq_f32(v1));
    }
    const float amax = vmaxvq_f32(vmax);
    if (!(amax > 0.f)) {
        memset(dt8, 0, (size_t) K);
        *inv_out = 0.f;
        return;
    }
    *inv_out = amax / 127.0f;
    const float32x4_t vsc = vdupq_n_f32(127.0f / amax);
    for (int k = 0; k < K; k += 16) {
        const int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(dt + k + 0), vsc));
        const int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(dt + k + 4), vsc));
        const int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(dt + k + 8), vsc));
        const int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(dt + k + 12), vsc));
        const int16x8_t p01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        const int16x8_t p23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        vst1q_s8(dt8 + k, vcombine_s8(vqmovn_s16(p01), vqmovn_s16(p23)));
    }
}

static void pq_s1_accumulate_subspace(const PQTensor & t, const pq_f16 * xh,
                                      int subspace, pq_f16 * yl) {
    const int ds = t.ds;
    const int K = t.K;
    const int dout = (int) t.n_out;
    const pq_f16 * cbh = t.cbh.data();
    const uint8_t * idx = t.idx.data();

    alignas(16) int8_t dt8[GGML_PQ_K];
    float inv = 0.f;
    if (ds == 4) {
        pq_build_quant_ds4(xh, cbh, subspace, K, dt8, &inv);
    } else {
        float dt[GGML_PQ_K];
        pq_build_dt_scalar(ds, xh, cbh, subspace, K, dt);
        pq_quantize_dt(dt, K, dt8, &inv);
    }
    const uint8_t * ii = idx + (size_t) subspace * dout;

    int j = 0;
    if (K == 64) {
        const PqLut64 L = pq_lut64_load(dt8);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        const float16_t hinv = (float16_t) inv;
        for (; j + 15 < dout; j += 16) {
            float16x8_t y0 = vld1q_f16((const __fp16 *) (yl + j));
            float16x8_t y1 = vld1q_f16((const __fp16 *) (yl + j + 8));
            const int8x16_t qv = pq_lut64_lookup(L, vld1q_u8(ii + j));
            float16x8_t lo, hi;
            pq_i8_to_f16x8x2(qv, lo, hi);
            y0 = vfmaq_n_f16(y0, lo, hinv);
            y1 = vfmaq_n_f16(y1, hi, hinv);
            vst1q_f16((__fp16 *) (yl + j), y0);
            vst1q_f16((__fp16 *) (yl + j + 8), y1);
        }
#else
        for (; j + 15 < dout; j += 16) {
            float tmp[16];
            for (int t_ = 0; t_ < 16; t_++) tmp[t_] = pq_to_f32(yl[j + t_]);
            float32x4_t y0 = vld1q_f32(tmp + 0), y1 = vld1q_f32(tmp + 4);
            float32x4_t y2 = vld1q_f32(tmp + 8), y3 = vld1q_f32(tmp + 12);
            const int8x16_t qv = pq_lut64_lookup(L, vld1q_u8(ii + j));
            const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
            const int16x8_t qhi = vmovl_high_s8(qv);
            y0 = vfmaq_n_f32(y0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))), inv);
            y1 = vfmaq_n_f32(y1, vcvtq_f32_s32(vmovl_high_s16(qlo)), inv);
            y2 = vfmaq_n_f32(y2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))), inv);
            y3 = vfmaq_n_f32(y3, vcvtq_f32_s32(vmovl_high_s16(qhi)), inv);
            vst1q_f32(tmp + 0, y0); vst1q_f32(tmp + 4, y1);
            vst1q_f32(tmp + 8, y2); vst1q_f32(tmp + 12, y3);
            for (int t_ = 0; t_ < 16; t_++) yl[j + t_] = pq_from_f32(tmp[t_]);
        }
#endif
    } else {
        const PqLut256 L = pq_lut256_load(dt8);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        const float16_t hinv = (float16_t) inv;
        for (; j + 15 < dout; j += 16) {
            if (j + 128 < dout) {
                pq_prefetch_t0(ii + j + 128);
            }
            float16x8_t y0 = vld1q_f16((const __fp16 *) (yl + j));
            float16x8_t y1 = vld1q_f16((const __fp16 *) (yl + j + 8));
            pq_fma_lut16_f16(y0, y1, L, ii + j, hinv);
            vst1q_f16((__fp16 *) (yl + j), y0);
            vst1q_f16((__fp16 *) (yl + j + 8), y1);
        }
#else
        for (; j + 15 < dout; j += 16) {
            if (j + 128 < dout) {
                pq_prefetch_t0(ii + j + 128);
            }
            float tmp[16];
            for (int t_ = 0; t_ < 16; t_++) {
                tmp[t_] = pq_to_f32(yl[j + t_]);
            }
            float32x4_t y0 = vld1q_f32(tmp + 0);
            float32x4_t y1 = vld1q_f32(tmp + 4);
            float32x4_t y2 = vld1q_f32(tmp + 8);
            float32x4_t y3 = vld1q_f32(tmp + 12);
            pq_fma_lut16_f32(y0, y1, y2, y3, L, ii + j, inv);
            vst1q_f32(tmp + 0, y0);
            vst1q_f32(tmp + 4, y1);
            vst1q_f32(tmp + 8, y2);
            vst1q_f32(tmp + 12, y3);
            for (int t_ = 0; t_ < 16; t_++) {
                yl[j + t_] = pq_from_f32(tmp[t_]);
            }
        }
#endif
    }
    for (; j < dout; j++) {
        yl[j] = pq_from_f32(pq_to_f32(yl[j]) + (float) dt8[ii[j]] * inv);
    }
}

// 4-subspace block: one yl tile load/store applies 4 LUTs (x86 ds4_block).
// Tile=64 keeps yl in regs; each LUT is loaded once per tile (4x reuse).
static void pq_s1_phase1_ds4_block(const PQTensor & t, const pq_f16 * xh,
                                   int i0, int i1, pq_f16 * yl) {
    const int K = t.K;
    if (K != GGML_PQ_K) {
        for (int i = i0; i < i1; i++) {
            pq_s1_accumulate_subspace(t, xh, i, yl);
        }
        return;
    }
    const int dout = (int) t.n_out;
    const pq_f16 * cbh = t.cbh.data();
    const uint8_t * idx = t.idx.data();
    int i = i0;
    for (; i + 3 < i1; i += 4) {
        if (i + 4 < i1) {
            pq_prefetch_t0(idx + (size_t) (i + 4) * dout);
            pq_prefetch_t0(cbh + (size_t) (i + 4) * 4 * K);
        }
        if (i + 8 < i1) {
            pq_prefetch_t1(idx + (size_t) (i + 8) * dout);
            pq_prefetch_t1(cbh + (size_t) (i + 8) * 4 * K);
        }

        alignas(16) int8_t dt8[4][GGML_PQ_K];
        float inv[4];
        for (int s = 0; s < 4; s++) {
            pq_build_quant_ds4(xh, cbh, i + s, K, dt8[s], &inv[s]);
        }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        const float16_t hinv[4] = {
            (float16_t) inv[0], (float16_t) inv[1],
            (float16_t) inv[2], (float16_t) inv[3],
        };
        int j = 0;
        for (; j + 63 < dout; j += 64) {
            if (j + 256 < dout) {
                pq_prefetch_t0(idx + (size_t) i * dout + j + 256);
            }
            float16x8_t y0 = vld1q_f16((const __fp16 *) (yl + j + 0));
            float16x8_t y1 = vld1q_f16((const __fp16 *) (yl + j + 8));
            float16x8_t y2 = vld1q_f16((const __fp16 *) (yl + j + 16));
            float16x8_t y3 = vld1q_f16((const __fp16 *) (yl + j + 24));
            float16x8_t y4 = vld1q_f16((const __fp16 *) (yl + j + 32));
            float16x8_t y5 = vld1q_f16((const __fp16 *) (yl + j + 40));
            float16x8_t y6 = vld1q_f16((const __fp16 *) (yl + j + 48));
            float16x8_t y7 = vld1q_f16((const __fp16 *) (yl + j + 56));
            for (int s = 0; s < 4; s++) {
                const PqLut256 L = pq_lut256_load(dt8[s]);
                const uint8_t * ii = idx + (size_t) (i + s) * dout + j;
                const float16_t iv = hinv[s];
                pq_fma_lut16_f16(y0, y1, L, ii + 0, iv);
                pq_fma_lut16_f16(y2, y3, L, ii + 16, iv);
                pq_fma_lut16_f16(y4, y5, L, ii + 32, iv);
                pq_fma_lut16_f16(y6, y7, L, ii + 48, iv);
            }
            vst1q_f16((__fp16 *) (yl + j + 0), y0);
            vst1q_f16((__fp16 *) (yl + j + 8), y1);
            vst1q_f16((__fp16 *) (yl + j + 16), y2);
            vst1q_f16((__fp16 *) (yl + j + 24), y3);
            vst1q_f16((__fp16 *) (yl + j + 32), y4);
            vst1q_f16((__fp16 *) (yl + j + 40), y5);
            vst1q_f16((__fp16 *) (yl + j + 48), y6);
            vst1q_f16((__fp16 *) (yl + j + 56), y7);
        }
        for (; j + 15 < dout; j += 16) {
            float16x8_t y0 = vld1q_f16((const __fp16 *) (yl + j));
            float16x8_t y1 = vld1q_f16((const __fp16 *) (yl + j + 8));
            for (int s = 0; s < 4; s++) {
                const PqLut256 L = pq_lut256_load(dt8[s]);
                pq_fma_lut16_f16(y0, y1, L,
                                 idx + (size_t) (i + s) * dout + j, hinv[s]);
            }
            vst1q_f16((__fp16 *) (yl + j), y0);
            vst1q_f16((__fp16 *) (yl + j + 8), y1);
        }
#else
        int j = 0;
        for (; j + 31 < dout; j += 32) {
            float tmp[32];
            for (int t = 0; t < 32; t++) {
                tmp[t] = pq_to_f32(yl[j + t]);
            }
            float32x4_t a0 = vld1q_f32(tmp + 0),  a1 = vld1q_f32(tmp + 4);
            float32x4_t a2 = vld1q_f32(tmp + 8),  a3 = vld1q_f32(tmp + 12);
            float32x4_t b0 = vld1q_f32(tmp + 16), b1 = vld1q_f32(tmp + 20);
            float32x4_t b2 = vld1q_f32(tmp + 24), b3 = vld1q_f32(tmp + 28);
            for (int s = 0; s < 4; s++) {
                const PqLut256 L = pq_lut256_load(dt8[s]);
                const uint8_t * ii = idx + (size_t) (i + s) * dout + j;
                pq_fma_lut16_f32(a0, a1, a2, a3, L, ii, inv[s]);
                pq_fma_lut16_f32(b0, b1, b2, b3, L, ii + 16, inv[s]);
            }
            vst1q_f32(tmp + 0, a0);  vst1q_f32(tmp + 4, a1);
            vst1q_f32(tmp + 8, a2);  vst1q_f32(tmp + 12, a3);
            vst1q_f32(tmp + 16, b0); vst1q_f32(tmp + 20, b1);
            vst1q_f32(tmp + 24, b2); vst1q_f32(tmp + 28, b3);
            for (int t = 0; t < 32; t++) {
                yl[j + t] = pq_from_f32(tmp[t]);
            }
        }
        for (; j + 15 < dout; j += 16) {
            float tmp[16];
            for (int t = 0; t < 16; t++) {
                tmp[t] = pq_to_f32(yl[j + t]);
            }
            float32x4_t y0 = vld1q_f32(tmp + 0), y1 = vld1q_f32(tmp + 4);
            float32x4_t y2 = vld1q_f32(tmp + 8), y3 = vld1q_f32(tmp + 12);
            for (int s = 0; s < 4; s++) {
                const PqLut256 L = pq_lut256_load(dt8[s]);
                pq_fma_lut16_f32(y0, y1, y2, y3, L,
                                 idx + (size_t) (i + s) * dout + j, inv[s]);
            }
            vst1q_f32(tmp + 0, y0); vst1q_f32(tmp + 4, y1);
            vst1q_f32(tmp + 8, y2); vst1q_f32(tmp + 12, y3);
            for (int t = 0; t < 16; t++) {
                yl[j + t] = pq_from_f32(tmp[t]);
            }
        }
#endif
        for (; j < dout; j++) {
            float v = pq_to_f32(yl[j]);
            for (int s = 0; s < 4; s++) {
                v += (float) dt8[s][idx[(size_t) (i + s) * dout + j]] * inv[s];
            }
            yl[j] = pq_from_f32(v);
        }
    }
    for (; i < i1; i++) {
        pq_s1_accumulate_subspace(t, xh, i, yl);
    }
}

static void pq_s1_phase1_ds4(const PQTensor & t, const pq_f16 * xh,
                             int i0, int i1, pq_f16 * yl) {
    static const bool g_generic = getenv("GGML_PQ_DS4_GENERIC") != nullptr;
    if (g_generic) {
        for (int i = i0; i < i1; i++) {
            pq_s1_accumulate_subspace(t, xh, i, yl);
        }
        return;
    }
    pq_s1_phase1_ds4_block(t, xh, i0, i1, yl);
}

static void pq_s1_phase1(const PQTensor & t, const pq_f16 * xh,
                         int i0, int i1, pq_f16 * yl) {
    if (t.ds == 4) {
        pq_s1_phase1_ds4(t, xh, i0, i1, yl);
        return;
    }
    for (int i = i0; i < i1; i++) {
        pq_s1_accumulate_subspace(t, xh, i, yl);
    }
}

static void pq_s1_phase2(const PQTensor & t, float * dst, int j0, int j1,
                         int nth, const pq_f16 * base) {
    const size_t row = (size_t) t.n_out;
    int j = j0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    for (; j + 15 < j1; j += 16) {
        float32x4_t s0 = vdupq_n_f32(0.f);
        float32x4_t s1 = vdupq_n_f32(0.f);
        float32x4_t s2 = vdupq_n_f32(0.f);
        float32x4_t s3 = vdupq_n_f32(0.f);
        for (int p = 0; p < nth; p++) {
            const pq_f16 * src = base + (size_t) p * row + j;
            const float16x8_t h0 = vld1q_f16((const __fp16 *) src);
            const float16x8_t h1 = vld1q_f16((const __fp16 *) (src + 8));
            s0 = vaddq_f32(s0, vcvt_f32_f16(vget_low_f16(h0)));
            s1 = vaddq_f32(s1, vcvt_f32_f16(vget_high_f16(h0)));
            s2 = vaddq_f32(s2, vcvt_f32_f16(vget_low_f16(h1)));
            s3 = vaddq_f32(s3, vcvt_f32_f16(vget_high_f16(h1)));
        }
        if (t.scaled) {
            s0 = vmulq_f32(s0, vld1q_f32(t.row_scale.data() + j + 0));
            s1 = vmulq_f32(s1, vld1q_f32(t.row_scale.data() + j + 4));
            s2 = vmulq_f32(s2, vld1q_f32(t.row_scale.data() + j + 8));
            s3 = vmulq_f32(s3, vld1q_f32(t.row_scale.data() + j + 12));
        }
        vst1q_f32(dst + j + 0, s0);
        vst1q_f32(dst + j + 4, s1);
        vst1q_f32(dst + j + 8, s2);
        vst1q_f32(dst + j + 12, s3);
    }
#endif
    for (; j + 7 < j1; j += 8) {
        float32x4_t sum0 = vdupq_n_f32(0.f);
        float32x4_t sum1 = vdupq_n_f32(0.f);
        for (int p = 0; p < nth; p++) {
            const pq_f16 * src = base + (size_t) p * row + j;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            const float16x8_t h = vld1q_f16((const __fp16 *) src);
            sum0 = vaddq_f32(sum0, vcvt_f32_f16(vget_low_f16(h)));
            sum1 = vaddq_f32(sum1, vcvt_f32_f16(vget_high_f16(h)));
#else
            float tmp[8] = {
                pq_to_f32(src[0]), pq_to_f32(src[1]), pq_to_f32(src[2]), pq_to_f32(src[3]),
                pq_to_f32(src[4]), pq_to_f32(src[5]), pq_to_f32(src[6]), pq_to_f32(src[7]),
            };
            sum0 = vaddq_f32(sum0, vld1q_f32(tmp));
            sum1 = vaddq_f32(sum1, vld1q_f32(tmp + 4));
#endif
        }
        if (t.scaled) {
            sum0 = vmulq_f32(sum0, vld1q_f32(t.row_scale.data() + j));
            sum1 = vmulq_f32(sum1, vld1q_f32(t.row_scale.data() + j + 4));
        }
        vst1q_f32(dst + j, sum0);
        vst1q_f32(dst + j + 4, sum1);
    }
    for (; j < j1; j++) {
        float sum = 0.f;
        for (int p = 0; p < nth; p++) {
            sum += pq_to_f32(base[(size_t) p * row + j]);
        }
        dst[j] = t.scaled ? sum * t.row_scale[(size_t) j] : sum;
    }
}

// Build dt8/inv for subspaces [i0,i1) into every NUMA replica (small write).
static void pq_s1_publish_luts(const PQTensor & t, const pq_f16 * xh,
                               int i0, int i1, int slot, int n_nodes) {
    const pq_f16 * cbh = t.cbh.data();
    const int ds = t.ds;
    const int K = t.K;
    for (int i = i0; i < i1; i++) {
        alignas(16) int8_t dt8[GGML_PQ_K];
        float inv = 0.f;
        if (ds == 4) {
            pq_build_quant_ds4(xh, cbh, i, K, dt8, &inv);
        } else {
            float dt[GGML_PQ_K];
            pq_build_dt_scalar(ds, xh, cbh, i, K, dt);
            pq_quantize_dt(dt, K, dt8, &inv);
        }
        for (int node = 0; node < n_nodes; node++) {
            memcpy(g_s1_luts[node][slot].dt8.data() + (size_t) i * K,
                   dt8, (size_t) K);
            g_s1_luts[node][slot].inv[(size_t) i] = inv;
        }
    }
}

// Output-owned accumulate over shared LUTs.
// Phase1-equivalent: int32 accumulate per subspace, then scale by inv once
// (avoids s8→fp per-FMA). Supports K=256 (4×tbl) and K=64 (1×tbl).
static void pq_s1_phase2_shared(const PQTensor & t, float * dst, int j0, int j1,
                                const int8_t * dt8, const float * inv) {
    const int M = (int) (t.n_in / t.ds);
    const int K = t.K;
    const int dout = (int) t.n_out;
    const uint8_t * idx = t.idx.data();

    for (int jbase = j0; jbase < j1; ) {
        const int jb = std::min(kS1OutTile, j1 - jbase);
        alignas(16) float facc[kS1OutTile];
        for (int z = 0; z < jb; z++) {
            facc[z] = 0.f;
        }

        for (int i = 0; i < M; i++) {
            if (i + 1 < M) {
                pq_prefetch_t0(dt8 + (size_t) (i + 1) * K);
                pq_prefetch_t0(idx + (size_t) (i + 1) * dout + jbase);
            }
            const float iv = inv[i];
            const uint8_t * ii = idx + (size_t) i * dout + jbase;
            const int8_t * row = dt8 + (size_t) i * K;

            alignas(16) int32_t iacc[kS1OutTile];
            for (int z = 0; z < jb; z++) {
                iacc[z] = 0;
            }

            if (K == 64) {
                const PqLut64 L = pq_lut64_load(row);
                int off = 0;
                for (; off + 15 < jb; off += 16) {
                    const int8x16_t qv = pq_lut64_lookup(L, vld1q_u8(ii + off));
                    const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
                    const int16x8_t qhi = vmovl_high_s8(qv);
                    int32x4_t a0 = vld1q_s32(iacc + off + 0);
                    int32x4_t a1 = vld1q_s32(iacc + off + 4);
                    int32x4_t a2 = vld1q_s32(iacc + off + 8);
                    int32x4_t a3 = vld1q_s32(iacc + off + 12);
                    a0 = vaddq_s32(a0, vmovl_s16(vget_low_s16(qlo)));
                    a1 = vaddq_s32(a1, vmovl_high_s16(qlo));
                    a2 = vaddq_s32(a2, vmovl_s16(vget_low_s16(qhi)));
                    a3 = vaddq_s32(a3, vmovl_high_s16(qhi));
                    vst1q_s32(iacc + off + 0, a0);
                    vst1q_s32(iacc + off + 4, a1);
                    vst1q_s32(iacc + off + 8, a2);
                    vst1q_s32(iacc + off + 12, a3);
                }
                for (; off < jb; off++) {
                    iacc[off] += (int32_t) row[ii[off]];
                }
            } else {
                const PqLut256 L = pq_lut256_load(row);
                int off = 0;
                for (; off + 15 < jb; off += 16) {
                    const int8x16_t qv = pq_lut256_lookup(L, vld1q_u8(ii + off));
                    const int16x8_t qlo = vmovl_s8(vget_low_s8(qv));
                    const int16x8_t qhi = vmovl_high_s8(qv);
                    int32x4_t a0 = vld1q_s32(iacc + off + 0);
                    int32x4_t a1 = vld1q_s32(iacc + off + 4);
                    int32x4_t a2 = vld1q_s32(iacc + off + 8);
                    int32x4_t a3 = vld1q_s32(iacc + off + 12);
                    a0 = vaddq_s32(a0, vmovl_s16(vget_low_s16(qlo)));
                    a1 = vaddq_s32(a1, vmovl_high_s16(qlo));
                    a2 = vaddq_s32(a2, vmovl_s16(vget_low_s16(qhi)));
                    a3 = vaddq_s32(a3, vmovl_high_s16(qhi));
                    vst1q_s32(iacc + off + 0, a0);
                    vst1q_s32(iacc + off + 4, a1);
                    vst1q_s32(iacc + off + 8, a2);
                    vst1q_s32(iacc + off + 12, a3);
                }
                for (; off < jb; off++) {
                    iacc[off] += (int32_t) row[ii[off]];
                }
            }

            int off = 0;
            for (; off + 3 < jb; off += 4) {
                float32x4_t f = vld1q_f32(facc + off);
                f = vfmaq_n_f32(f, vcvtq_f32_s32(vld1q_s32(iacc + off)), iv);
                vst1q_f32(facc + off, f);
            }
            for (; off < jb; off++) {
                facc[off] += (float) iacc[off] * iv;
            }
        }

        for (int u = 0; u < jb; u++) {
            float v = facc[u];
            if (t.scaled) {
                v *= t.row_scale[(size_t) (jbase + u)];
            }
            dst[jbase + u] = v;
        }
        jbase += jb;
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
    (void) xb.sx;

    for (int i = i0; i < i1; i++) {
        if (i + 1 < i1) {
            pq_prefetch_t0(idx + (size_t) (i + 1) * din);
            pq_prefetch_t0(cb8 + (size_t) (i + 1) * 512);
        }
        if (i + 8 < i1) {
            pq_prefetch_t1(idx + (size_t) (i + 8) * din);
            pq_prefetch_t1(cb8 + (size_t) (i + 8) * 512);
        }

        const int8_t * ce = cb8 + (size_t) i * 512;
        const int8_t * co = ce + 256;
        const uint8_t * ii = idx + (size_t) i * din;
        const PqLut256 Le = pq_lut256_load(ce);
        const PqLut256 Lo = pq_lut256_load(co);

        int32x4_t ae = vdupq_n_s32(0);
        int32x4_t ao = vdupq_n_s32(0);
        int j = 0;
        for (; j + 15 < din; j += 16) {
            if (j + 128 < din) {
                pq_prefetch_t0(ii + j + 128);
                pq_prefetch_t0(xq + j + 128);
            }
            const int8x16_t xv = vld1q_s8(xq + j);
            const uint8x16_t iv = vld1q_u8(ii + j);
            const int8x16_t qe = pq_lut256_lookup(Le, iv);
            const int8x16_t qo = pq_lut256_lookup(Lo, iv);
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
    const int K = t.K;
    const int din = (int) t.n_in;
    const int8_t * cb8 = t.cb8.data();
    const float * inv = t.inv.data();
    const uint8_t * idx = t.idx.data();
    const pq_f16 * xh = xb.xh;

    for (int i = i0; i < i1; i++) {
        const int8_t * cbi = cb8 + (size_t) i * ds * K;
        const uint8_t * ii = idx + (size_t) i * din;
        for (int s = 0; s < ds; s++) {
            float sum = 0.f;
            for (int j = 0; j < din; j++) {
                sum += pq_to_f32(xh[j]) * cbi[(size_t) s * K + ii[j]];
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
    if (t.ds == 2 && t.K == GGML_PQ_K) {
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

// Per-OS-thread group state: avoids a fixed ith cap (Kunpeng can be 640+ CPUs)
// and matches OpenMP's one-worker-per-thread scheduling.
static thread_local S1Group g_s1_tls;

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

static void pq_s1_flush_partials(S1Group & g, int ith, int nth, void * threadpool,
                                 bool sync) {
    const int n = g.n;
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
        memset(my, 0, (size_t) tm.n_out * sizeof(pq_f16));
        if (a1 > a0) {
            pq_s1_phase1(tm, g.xh[m], a0, a1, my);
        }
    }

    pq_barrier(threadpool, nth);

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
        pq_barrier(threadpool, nth);
    }
}

// Default Kunpeng path: shared LUT + output-owned (no O(nth x n_out) partials).
static void pq_s1_flush_shared(S1Group & g, int ith, int nth, void * threadpool,
                               bool sync) {
    const int n = g.n;
    const int n_nodes = pq_numa_n_nodes();
    int Dtot = 0;
    for (int m = 0; m < n; m++) {
        Dtot += (int) g.t[m]->n_out;
    }

    // Grow scratch on thread 0 only, then barrier so no publish races a resize.
    if (ith == 0) {
        std::lock_guard<std::mutex> lk(g_s1_mu);
        for (int m = 0; m < n; m++) {
            const int M = (int) (g.t[m]->n_in / g.t[m]->ds);
            pq_s1_ensure_luts(n_nodes, m, M, g.t[m]->K);
        }
    }
    pq_barrier(threadpool, nth);

    for (int m = 0; m < n; m++) {
        const PQTensor & tm = *g.t[m];
        const int M = (int) (tm.n_in / tm.ds);
        const int a0 = (int) ((int64_t) M * ith / nth);
        const int a1 = (int) ((int64_t) M * (ith + 1) / nth);
        if (a1 > a0) {
            pq_s1_publish_luts(tm, g.xh[m], a0, a1, m, n_nodes);
        }
    }

    // LUTs published; output-owned phase.
    pq_barrier(threadpool, nth);

    const int node = pq_numa_node();
    const int j0 = (int) ((int64_t) Dtot * ith / nth);
    const int j1 = (int) ((int64_t) Dtot * (ith + 1) / nth);
    int d0 = 0;
    for (int m = 0; m < n; m++) {
        const PQTensor & tm = *g.t[m];
        const int b0 = std::max(j0, d0);
        const int b1 = std::min(j1, d0 + (int) tm.n_out);
        if (b1 > b0) {
            const S1LutScratch & lut = g_s1_luts[node][m];
            pq_s1_phase2_shared(tm, g.dst[m], b0 - d0, b1 - d0,
                                lut.dt8.data(), lut.inv.data());
        }
        d0 += (int) tm.n_out;
    }
    if (sync) {
        pq_barrier(threadpool, nth);
    }
}

static void pq_s1_flush(S1Group & g, int ith, int nth, void * threadpool, bool sync) {
    if (!g.active) {
        return;
    }
    g.active = false;
    const int n = g.n;

    const uint64_t tq0 = (ith == 0) ? ggml_time_us() : 0;

    if (pq_s1_use_partials()) {
        pq_s1_flush_partials(g, ith, nth, threadpool, sync);
    } else {
        pq_s1_flush_shared(g, ith, nth, threadpool, sync);
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
    g_s1_partials.clear();
    for (int node = 0; node < kMaxNumaNodes; node++) {
        for (int s = 0; s < 4; s++) {
            g_s1_luts[node][s].dt8.clear();
            g_s1_luts[node][s].inv.clear();
        }
    }
}

void ggml_pq_set_enabled(bool v) { g_pq_enabled.store(v); }
bool ggml_pq_enabled(void) { return g_pq_enabled.load(); }

bool ggml_pq_register(const char * name, int mode, int ds,
                      const float * cb_f32, const int8_t * cb8, const float * inv,
                      const uint8_t * idx, int64_t n_in, int64_t n_out, int K) {
    if (!name || ds < 1 || ds > GGML_PQ_MAX_DS || (K != GGML_PQ_K && K != GGML_PQ_K_ARM)) {
        return false;
    }
    PQTensor t;
    t.name = name;
    t.mode = mode;
    t.ds = ds;
    t.K = K;
    t.n_in = n_in;
    t.n_out = n_out;
    if (mode == 0) {
        if (!cb_f32 || !idx) {
            return false;
        }
        std::vector<pq_f16> tmp((size_t) (n_in / ds) * ds * K);
        for (int64_t i = 0; i < n_in / ds; i++) {
            for (int d = 0; d < ds; d++) {
                for (int k = 0; k < K; k++) {
                    tmp[((size_t) (i * ds + d) * K) + k] =
                        pq_from_f32(cb_f32[((size_t) i * K + k) * ds + d]);
                }
            }
        }
        t.cbh.assign(tmp.data(), tmp.size());
        t.idx.assign(idx, (size_t) (n_in / ds) * n_out);
    } else {
        if (!cb8 || !inv || !idx) {
            return false;
        }
        t.cb8.assign(cb8, (size_t) (n_out / ds) * ds * K);
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
                          int64_t n_in, int64_t n_out, int K) {
    if (!name || ds < 1 || ds > GGML_PQ_MAX_DS || !cb || !idx ||
        (K != GGML_PQ_K && K != GGML_PQ_K_ARM)) {
        return false;
    }
    PQTensor t;
    t.name = name;
    t.mode = mode;
    t.ds = ds;
    t.K = K;
    t.n_in = n_in;
    t.n_out = n_out;
    if (mode == 0) {
        t.cbh.assign((const pq_f16 *) cb, (size_t) (n_in / ds) * ds * K);
        t.idx.assign(idx, (size_t) (n_in / ds) * n_out);
    } else {
        t.cb8.assign((const int8_t *) cb, (size_t) (n_out / ds) * ds * K);
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
                                 int64_t n_in, int64_t n_out, int K) {
    if (!name || ds < 1 || ds > GGML_PQ_MAX_DS || !cb || !idx || !row_scale ||
        n_in % ds != 0 || (K != GGML_PQ_K && K != GGML_PQ_K_ARM)) {
        return false;
    }
    PQTensor t;
    t.name = name;
    t.mode = 0;
    t.ds = ds;
    t.K = K;
    t.n_in = n_in;
    t.n_out = n_out;
    t.cbh.assign((const pq_f16 *) cb, (size_t) n_in * K);
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
        S1Group & g = g_s1_tls;
        pq_s1_flush(g, ith, nth, threadpool, true);
        for (int m = 0; m < t.n_seg; m++) {
            pq_s1_group_add(g, *t.seg_t[m], key, x, x_type, dst + t.seg_off[m]);
        }
        pq_s1_flush(g, ith, nth, threadpool, false);
        return GGML_PQ_MM_DONE;
    }

    S1Group & g = g_s1_tls;
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
    const struct ggml_tensor * node = (const struct ggml_tensor *) node_ptr;
    if (node && pq_node_is_decode_mul_mat(node) && g_pq.count(node->src[0]->name)) {
        return;
    }
    pq_s1_flush(g_s1_tls, ith, nth, threadpool, true);
}

void ggml_pq_graph_end(int ith, int nth, void * threadpool) {
    if (!g_pq_enabled.load(std::memory_order_relaxed)) {
        return;
    }
    pq_s1_flush(g_s1_tls, ith, nth, threadpool, true);
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
