// Numeric self-test for the PQ decode GEMV path (incl. same-input fusion).
// Builds a tiny graph with two decode-shaped MUL_MAT nodes sharing one input,
// compares PQ outputs against an fp32 reference.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-pq.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const int n_in  = argc > 2 ? atoi(argv[2]) : 512;
    const int n_out = argc > 3 ? atoi(argv[3]) : 256;
    const int ds    = 2;
    const int M     = n_in / ds;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::uniform_int_distribution<int> ui(0, GGML_PQ_K - 1);

    // two S1 tensors sharing one input (fusion group of 2)
    const int NT = 2;
    std::vector<std::vector<_Float16>> cb(NT);
    std::vector<std::vector<uint8_t>>  idx(NT);
    std::vector<float> x(n_in);
    for (auto & v : x) v = u(rng);
    for (int t = 0; t < NT; t++) {
        cb[t].resize((size_t) M * ds * GGML_PQ_K);
        for (auto & v : cb[t]) v = (_Float16) u(rng);
        idx[t].resize((size_t) M * n_out);
        for (auto & v : idx[t]) v = (uint8_t) ui(rng);
    }

    ggml_pq_reset();
    for (int t = 0; t < NT; t++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "test.w%d", t);
        if (!ggml_pq_register_raw(nm, 0, ds, cb[t].data(), nullptr, idx[t].data(),
                                  n_in, n_out)) {
            fprintf(stderr, "register failed\n");
            return 1;
        }
    }
    ggml_pq_set_enabled(true);

    // fp32 reference
    std::vector<std::vector<float>> ref(NT, std::vector<float>(n_out, 0.f));
    for (int t = 0; t < NT; t++)
        for (int i = 0; i < M; i++) {
            float dt[GGML_PQ_K];
            for (int k = 0; k < GGML_PQ_K; k++) {
                float s = 0.f;
                for (int d = 0; d < ds; d++)
                    s += x[(size_t) i * ds + d] * (float) cb[t][((size_t)(i * ds + d)) * GGML_PQ_K + k];
                dt[k] = s;
            }
            for (int j = 0; j < n_out; j++)
                ref[t][j] += dt[idx[t][(size_t) i * n_out + j]];
        }

    // graph: x(1,n_in) -> w0*x, w1*x (two MUL_MAT, same src1)
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
    // consumer node: forces a mid-graph boundary flush before it computes
    struct ggml_tensor * zsum = ggml_add(ctx, y0, y1);
    ggml_set_name(zsum, "zsum");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, zsum);

    const int nth = argc > 1 ? atoi(argv[1]) : 8;
    struct ggml_cplan plan = ggml_graph_plan(gf, nth, nullptr);
    plan.work_data = (uint8_t *) malloc(plan.work_size);
    if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }

    int bad = 0;
    const struct ggml_tensor * ys[NT] = { y0, y1 };
    {
        // consumer node must observe both flushed outputs
        const float * z = (const float *) zsum->data;
        double se = 0, sr = 0;
        for (int j = 0; j < n_out; j++) {
            const double r = ref[0][j] + ref[1][j];
            const double e = z[j] - r;
            se += e * e; sr += r * r;
        }
        printf("zsum: rel_mse=%.3e\n", sr > 0 ? se / sr : 0.0);
        if (!(sr > 0 ? se / sr : 0.0) > 0.05) bad = 1;
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
        printf("w%d: rel_mse=%.3e  y[0..3]=%.3f %.3f %.3f %.3f  ref=%.3f %.3f %.3f %.3f\n",
               t, rel, y[0], y[1], y[2], y[3], ref[t][0], ref[t][1], ref[t][2], ref[t][3]);
        if (rel > 0.05) {
            for (int j = 0; j < n_out; j += 32)
                printf("  j=%3d y=%8.3f ref=%8.3f\n", j, y[j], ref[t][j]);
        }
        if (rel > 0.05 || !std::isfinite(y[0])) bad = 1;
    }
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
