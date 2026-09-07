// Numeric self-test for the PQ decode GEMV path (S1 fusion + S2 direct GEMV).
// Compares PQ outputs against an fp32 reference on the active CPU backend
// (AVX-512 on x86, NEON on aarch64 / Kunpeng).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-pq.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int test_s1_fusion(int nth, int n_in, int n_out) {
    const int ds = 2;
    const int M  = n_in / ds;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::uniform_int_distribution<int> ui(0, GGML_PQ_K - 1);

    const int NT = 2;
    std::vector<std::vector<ggml_fp16_t>> cb(NT);
    std::vector<std::vector<uint8_t>>  idx(NT);
    std::vector<float> x(n_in);
    for (auto & v : x) v = u(rng);
    for (int t = 0; t < NT; t++) {
        cb[t].resize((size_t) M * ds * GGML_PQ_K);
        for (auto & v : cb[t]) v = ggml_fp32_to_fp16(u(rng));
        idx[t].resize((size_t) M * n_out);
        for (auto & v : idx[t]) v = (uint8_t) ui(rng);
    }

    ggml_pq_reset();
    for (int t = 0; t < NT; t++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "test.w%d", t);
        if (!ggml_pq_register_raw(nm, /*mode=*/0, ds, cb[t].data(), nullptr,
                                  idx[t].data(), n_in, n_out)) {
            fprintf(stderr, "S1 register failed\n");
            return 1;
        }
    }
    ggml_pq_set_enabled(true);

    std::vector<std::vector<float>> ref(NT, std::vector<float>(n_out, 0.f));
    for (int t = 0; t < NT; t++) {
        for (int i = 0; i < M; i++) {
            float dt[GGML_PQ_K];
            for (int k = 0; k < GGML_PQ_K; k++) {
                float s = 0.f;
                for (int d = 0; d < ds; d++) {
                    s += x[(size_t) i * ds + d] *
                         (float) cb[t][((size_t)(i * ds + d)) * GGML_PQ_K + k];
                }
                dt[k] = s;
            }
            for (int j = 0; j < n_out; j++) {
                ref[t][j] += dt[idx[t][(size_t) i * n_out + j]];
            }
        }
    }

    struct ggml_init_params ip = { 256 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * w0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_in, n_out);
    ggml_set_name(w0, "test.w0");
    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_in, n_out);
    ggml_set_name(w1, "test.w1");
    struct ggml_tensor * xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, 1);
    memcpy(xin->data, x.data(), n_in * sizeof(float));

    struct ggml_tensor * y0 = ggml_mul_mat(ctx, w0, xin);
    ggml_set_name(y0, "y0");
    struct ggml_tensor * y1 = ggml_mul_mat(ctx, w1, xin);
    ggml_set_name(y1, "y1");
    struct ggml_tensor * zsum = ggml_add(ctx, y0, y1);
    ggml_set_name(zsum, "zsum");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, zsum);

    struct ggml_cplan plan = ggml_graph_plan(gf, nth, nullptr);
    plan.work_data = (uint8_t *) malloc(plan.work_size);
    if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "S1 compute failed\n");
        return 1;
    }

    int bad = 0;
    const struct ggml_tensor * ys[NT] = { y0, y1 };
    {
        const float * z = (const float *) zsum->data;
        double se = 0, sr = 0;
        for (int j = 0; j < n_out; j++) {
            const double r = ref[0][j] + ref[1][j];
            const double e = z[j] - r;
            se += e * e;
            sr += r * r;
        }
        const double rel = sr > 0 ? se / sr : 0.0;
        printf("S1 zsum: rel_mse=%.3e\n", rel);
        if (rel > 0.05) {
            bad = 1;
        }
    }
    for (int t = 0; t < NT; t++) {
        const float * y = (const float *) ys[t]->data;
        double se = 0, sr = 0;
        for (int j = 0; j < n_out; j++) {
            const double e = y[j] - ref[t][j];
            se += e * e;
            sr += (double) ref[t][j] * ref[t][j];
        }
        const double rel = sr > 0 ? se / sr : 0;
        printf("S1 w%d: rel_mse=%.3e\n", t, rel);
        if (rel > 0.05 || !std::isfinite(y[0])) {
            bad = 1;
        }
    }
    free(plan.work_data);
    ggml_free(ctx);
    return bad;
}

// Direct S2 GEMV vs fp32 reference (covers NEON 256-entry LUT on aarch64).
static int test_s2_direct(int nth, int n_in, int n_out) {
    const int ds = 2;
    if (n_out % ds != 0) {
        fprintf(stderr, "S2 n_out must be divisible by ds\n");
        return 1;
    }
    const int M = n_out / ds;

    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::uniform_int_distribution<int> ui(0, GGML_PQ_K - 1);
    std::uniform_int_distribution<int> qi(-127, 127);

    std::vector<int8_t> cb8((size_t) M * ds * GGML_PQ_K);
    std::vector<float> inv((size_t) M * ds);
    std::vector<uint8_t> idx((size_t) M * n_in);
    std::vector<float> x(n_in);
    for (auto & v : x) {
        v = u(rng);
    }
    for (auto & v : cb8) {
        v = (int8_t) qi(rng);
    }
    for (auto & v : inv) {
        v = 0.01f + 0.05f * std::fabs(u(rng));
    }
    for (auto & v : idx) {
        v = (uint8_t) ui(rng);
    }

    ggml_pq_reset();
    if (!ggml_pq_register_raw("test.s2", /*mode=*/1, ds, cb8.data(), inv.data(),
                              idx.data(), n_in, n_out)) {
        fprintf(stderr, "S2 register failed\n");
        return 1;
    }
    ggml_pq_set_enabled(true);

    std::vector<float> ref(n_out, 0.f);
    for (int i = 0; i < M; i++) {
        for (int s = 0; s < ds; s++) {
            float sum = 0.f;
            const int8_t * cbi = cb8.data() + ((size_t) i * ds + s) * GGML_PQ_K;
            const uint8_t * ii = idx.data() + (size_t) i * n_in;
            for (int j = 0; j < n_in; j++) {
                sum += (x[j] / 128.0f) * (float) cbi[ii[j]];
            }
            ref[(size_t) i * ds + s] = sum * (inv[(size_t) i * ds + s] * 128.0f);
        }
    }

    std::vector<float> y(n_out, 0.f);
    for (int ith = 0; ith < nth; ith++) {
        if (!ggml_pq_mul_mat_vec("test.s2", x.data(), /*F32*/0, y.data(),
                                 n_in, n_out, ith, nth, nullptr)) {
            fprintf(stderr, "S2 mul_mat_vec failed\n");
            return 1;
        }
    }

    double se = 0, sr = 0;
    for (int j = 0; j < n_out; j++) {
        const double e = y[j] - ref[j];
        se += e * e;
        sr += (double) ref[j] * ref[j];
    }
    const double rel = sr > 0 ? se / sr : 0;
    printf("S2: rel_mse=%.3e  y[0..3]=%.3f %.3f %.3f %.3f  ref=%.3f %.3f %.3f %.3f\n",
           rel, y[0], y[1], y[2], y[3], ref[0], ref[1], ref[2], ref[3]);
    if (rel > 0.05 || !std::isfinite(y[0])) {
        return 1;
    }
    return 0;
}

int main(int argc, char ** argv) {
    const int nth   = argc > 1 ? atoi(argv[1]) : 8;
    const int n_in  = argc > 2 ? atoi(argv[2]) : 512;
    const int n_out = argc > 3 ? atoi(argv[3]) : 256;

#if defined(__aarch64__)
    printf("pq-selftest: aarch64 / Kunpeng NEON path\n");
#elif defined(__AVX512FP16__) && defined(__AVX512VBMI__)
    printf("pq-selftest: x86 AVX-512 FP16+VBMI path\n");
#else
    printf("pq-selftest: host has no optimized PQ ISA; stubs may no-op\n");
#endif

    int bad = 0;
    bad |= test_s1_fusion(nth, n_in, n_out);
    bad |= test_s2_direct(nth, n_in, n_out);
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
