// llama-pq-convert: append PQ (product quantization) side tensors to a GGUF
// model so that --pq-decode models load instantly (no k-means at load time).
//
// For each quantized linear weight W ("blk.N.attn_q.weight" etc.) the output
// GGUF additionally contains:
//   <name>.pq_meta : I32 [4]         = { mode, ds, K, n_subspaces }
//   <name>.pq_cb   : S1 F32 [ds,K,M] | S2 I8 [K,M*ds]
//   <name>.pq_idx  : I8  [n_out,M] (S1) | [n_in,M] (S2)
//   <name>.pq_inv  : S2 F32 [M*ds]
// The original weights are kept untouched (prefill uses them).
// With --train-from <f16.gguf>, base weights come from <input.gguf> (e.g. Q8_0,
// used at prefill) while PQ codebooks/indices are trained on the F16 weights.
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-pq.h"
#include "llama-pq.h"

// (build info not needed)

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

namespace {

bool is_pq_candidate(const char * name, int64_t ne0, int64_t ne1) {
    if (ne0 < 256 || ne1 < 256) return false;
    static const char * keys[] = {
        ".attn_q.", ".attn_k.", ".attn_v.", ".attn_o.", ".attn_output.",
        ".ffn_gate.", ".ffn_down.", ".ffn_up.",
    };
    for (auto k : keys) {
        if (strstr(name, k)) return true;
    }
    return false;
}

// relative MSE of the packed representation vs the actual (dequantized) weight
static double pack_rel_mse(const struct ggml_tensor * t, const llama_pq_pack & p) {
    const int64_t n_out = t->ne[1], n_in = t->ne[0];
    std::vector<float> W((size_t) n_out * n_in);
    ggml_get_type_traits(t->type)->to_float(t->data, W.data(), n_out * n_in);
    double se = 0, se_w = 0;
    if (p.mode == 0) {
        const int M = (int)(n_in / p.ds);
        for (int64_t j = 0; j < n_out; j++) {
            for (int i = 0; i < M; i++) {
                // idx layout is subspace-major: idx[i * n_out + j]
                const float * cb = &p.cbf[((size_t) i * GGML_PQ_K + p.idx[(size_t) i * n_out + j]) * p.ds];
                for (int d = 0; d < p.ds; d++) {
                    const float e = W[(size_t) j * n_in + i * p.ds + d] - cb[d];
                    se += (double) e * e;
                }
            }
        }
    } else {
        const int M = (int)(n_out / p.ds);
        for (int64_t pw = 0; pw < n_in; pw++) {
            for (int i = 0; i < M; i++) {
                const uint8_t id = p.idx[(size_t) i * n_in + pw];
                for (int s = 0; s < p.ds; s++) {
                    const float v = p.inv[(size_t) i * p.ds + s] *
                                    p.cb8[((size_t) i * p.ds + s) * GGML_PQ_K + id];
                    const float e = W[(size_t)(i * p.ds + s) * n_in + pw] - v;
                    se += (double) e * e;
                }
            }
        }
    }
    for (size_t k = 0; k < W.size(); k++) se_w += (double) W[k] * W[k];
    return se_w > 0 ? se / se_w : 0.0;
}

// create a metadata-only ggml tensor and point it at `data`
struct ggml_tensor * make_tensor(struct ggml_context * ctx, const char * name,
                                 ggml_type type, int ndim, const int64_t * ne,
                                 void * data) {
    struct ggml_tensor * t = ggml_new_tensor(ctx, type, ndim, ne);
    ggml_set_name(t, name);
    t->data = data;
    return t;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.gguf> <output.gguf> [--pq-ds N] [--pq-mode auto|s1|s2]"
                        " [--train-from <f16.gguf>] [--pq4c8b-dir <dir>]\n", argv[0]);
        return 1;
    }
    const char * in_path  = argv[1];
    const char * out_path = argv[2];
    const char * tr_path  = nullptr;
    const char * pq4_dir  = nullptr;
    int ds = 2;
    int mode_cfg = 0; // 0=auto, 1=s1, 2=s2
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--train-from") && i + 1 < argc) tr_path = argv[++i];
        else if (!strcmp(argv[i], "--pq4c8b-dir") && i + 1 < argc) pq4_dir = argv[++i];
        else if (!strncmp(argv[i], "--pq-ds=", 8))      ds = atoi(argv[i] + 8);
        else if (!strcmp(argv[i], "--pq-ds") && i + 1 < argc) ds = atoi(argv[++i]);
        else if (!strncmp(argv[i], "--pq-mode=", 10)) {
            const char * m = argv[i] + 10;
            mode_cfg = !strcmp(m, "s1") ? 1 : (!strcmp(m, "s2") ? 2 : 0);
        } else if (!strcmp(argv[i], "--pq-mode") && i + 1 < argc) {
            const char * m = argv[++i];
            mode_cfg = !strcmp(m, "s1") ? 1 : (strcmp(m, "s2") ? 0 : 2);
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 1;
        }
    }
    if (ds < 1 || ds > GGML_PQ_MAX_DS) {
        fprintf(stderr, "invalid --pq-ds %d\n", ds);
        return 1;
    }

    fprintf(stderr, "%s: loading %s\n", __func__, in_path);
    struct gguf_init_params ip = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ nullptr,
    };
    struct ggml_context * ctx_in_data = nullptr;
    ip.ctx = &ctx_in_data;
    struct gguf_context * ctx_in_meta = gguf_init_from_file(in_path, ip);
    if (!ctx_in_meta || !ctx_in_data) {
        fprintf(stderr, "%s: failed to load %s\n", __func__, in_path);
        return 1;
    }

    // optional separate training source (e.g. F16 original weights)
    struct ggml_context * ctx_tr_data = nullptr;
    struct gguf_context * ctx_tr_meta = nullptr;
    if (tr_path) {
        fprintf(stderr, "%s: loading training weights from %s\n", __func__, tr_path);
        ip.ctx = &ctx_tr_data;
        ctx_tr_meta = gguf_init_from_file(tr_path, ip);
        if (!ctx_tr_meta || !ctx_tr_data) {
            fprintf(stderr, "%s: failed to load %s\n", __func__, tr_path);
            return 1;
        }
    }

    struct gguf_context * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in_meta);
    gguf_set_val_u32(ctx_out, "pq.ds", ds);
    gguf_set_val_u32(ctx_out, "pq.mode", mode_cfg);

    const int n_tensors = gguf_get_n_tensors(ctx_in_meta);
    int n_pq = 0;
    // side-tensor storage must outlive gguf_write_to_file
    std::vector<std::vector<char>> storages;
    struct ggml_context * ctx_side = nullptr;
    {
        struct ggml_init_params gip = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(n_tensors * 4 + 16),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_side = ggml_init(gip);
    }

    for (int ti = 0; ti < n_tensors; ti++) {
        const char * name = gguf_get_tensor_name(ctx_in_meta, ti);
        struct ggml_tensor * t = ggml_get_tensor(ctx_in_data, name);
        if (!t) continue;
        if (strstr(name, ".pq_meta") || strstr(name, ".pq_cb") ||
            strstr(name, ".pq_idx") || strstr(name, ".pq_inv") ||
            strstr(name, ".pq_row_scale")) continue;
        gguf_add_tensor(ctx_out, t);

        if (t->ne[2] != 1 || t->ne[3] != 1) continue;
        if (!is_pq_candidate(name, t->ne[0], t->ne[1])) continue;

        int mode = mode_cfg == 1 ? 0 : (mode_cfg == 2 ? 1 : (strstr(name, ".ffn_down.") ? 1 : 0));
        // train on the dedicated source tensor when given, else on the base tensor
        struct ggml_tensor * t_tr = t;
        if (ctx_tr_data) {
            t_tr = ggml_get_tensor(ctx_tr_data, name);
            if (!t_tr) {
                fprintf(stderr, "%s: training tensor '%s' not found, skipping\n", __func__, name);
                continue;
            }
            if (t_tr->ne[0] != t->ne[0] || t_tr->ne[1] != t->ne[1]) {
                fprintf(stderr, "%s: shape mismatch for '%s' (base %lldx%lld vs train %lldx%lld), skipping\n",
                        __func__, name, (long long) t->ne[0], (long long) t->ne[1],
                        (long long) t_tr->ne[0], (long long) t_tr->ne[1]);
                continue;
            }
        }
        char nm[512];
        if (pq4_dir) {
            const std::string stem = std::string(pq4_dir) + "/" + name;
            std::ifstream fcb(stem + ".cb", std::ios::binary), fidx(stem + ".idx", std::ios::binary), fsc(stem + ".scale", std::ios::binary);
            const size_t cb_bytes = (size_t)t->ne[0] * GGML_PQ_K * sizeof(uint16_t);
            const size_t idx_bytes = (size_t)(t->ne[0] / 4) * t->ne[1];
            const size_t sc_bytes = (size_t)t->ne[1] * sizeof(float);
            if (fcb && fidx && fsc) {
                std::vector<char> cb(cb_bytes), idx(idx_bytes), scale(sc_bytes);
                fcb.read(cb.data(), cb.size()); fidx.read(idx.data(), idx.size()); fsc.read(scale.data(), scale.size());
                if ((size_t)fcb.gcount() == cb.size() && (size_t)fidx.gcount() == idx.size() && (size_t)fsc.gcount() == scale.size()) {
                    const int32_t meta[4] = { 4, 4, GGML_PQ_K, (int32_t)(t->ne[0] / 4) };
                    storages.emplace_back(sizeof(meta)); memcpy(storages.back().data(), meta, sizeof(meta));
                    llama_pq_side_name(nm, sizeof(nm), name, "pq_meta");
                    const int64_t ne_meta[1] = { 4 };
                    gguf_add_tensor(ctx_out, make_tensor(ctx_side, nm, GGML_TYPE_I32, 1, ne_meta, storages.back().data()));
                    auto add_raw = [&](const char * suffix, ggml_type type, int ndim, const int64_t * ne, const std::vector<char> & bytes) {
                        storages.push_back(bytes); llama_pq_side_name(nm, sizeof(nm), name, suffix);
                        auto * st = ggml_new_tensor(ctx_side, type, ndim, ne); ggml_set_name(st, nm); st->data = storages.back().data(); gguf_add_tensor(ctx_out, st);
                    };
                    const int64_t ne_cb[2] = { GGML_PQ_K, t->ne[0] };
                    const int64_t ne_idx[2] = { t->ne[1], t->ne[0] / 4 };
                    const int64_t ne_sc[1] = { t->ne[1] };
                    add_raw("pq_cb", GGML_TYPE_F16, 2, ne_cb, cb);
                    add_raw("pq_idx", GGML_TYPE_I8, 2, ne_idx, idx);
                    add_raw("pq_row_scale", GGML_TYPE_F32, 1, ne_sc, scale);
                    n_pq++;
                    fprintf(stderr, "%s: PQ-4c8b packed '%s'\n", __func__, name);
                    continue;
                }
            }
        }
        llama_pq_pack pack;
        if (!llama_pq_build_pack(t_tr, mode, ds, pack)) continue;

        // meta = { mode, ds, K, M }
        const int32_t meta[4] = { mode, ds, GGML_PQ_K,
                                  (int32_t)(mode == 0 ? t->ne[0] / ds : t->ne[1] / ds) };
        storages.emplace_back(sizeof(meta));
        memcpy(storages.back().data(), meta, sizeof(meta));
        llama_pq_side_name(nm, sizeof(nm), name, "pq_meta");
        const int64_t ne_meta[1] = { 4 };
        gguf_add_tensor(ctx_out, make_tensor(ctx_side, nm, GGML_TYPE_I32, 1, ne_meta,
                                             storages.back().data()));

        auto add_side = [&](const char * suffix, ggml_type type, int ndim,
                            const int64_t * ne, const void * src, size_t bytes) {
            storages.emplace_back(bytes);
            memcpy(storages.back().data(), src, bytes);
            llama_pq_side_name(nm, sizeof(nm), name, suffix);
            struct ggml_tensor * st = ggml_new_tensor(ctx_side, type, ndim, ne);
            ggml_set_name(st, nm);
            st->data = storages.back().data();
            gguf_add_tensor(ctx_out, st);
        };

        if (mode == 0) {
            // S1: cb F32 [ds, K, M] (layout [(i*K+k)*ds+d]); idx I8 [n_out, M]
            const int M = (int)(t->ne[0] / ds);
            const int64_t ne_cb[3] = { ds, GGML_PQ_K, M };
            add_side("pq_cb", GGML_TYPE_F32, 3, ne_cb, pack.cbf.data(),
                     pack.cbf.size() * sizeof(float));
            const int64_t ne_idx[2] = { t->ne[1], M };
            add_side("pq_idx", GGML_TYPE_I8, 2, ne_idx, pack.idx.data(), pack.idx.size());
        } else {
            // S2: cb I8 [K, M*ds]; inv F32 [M*ds]; idx I8 [n_in, M]
            const int M = (int)(t->ne[1] / ds);
            const int64_t ne_cb[2] = { GGML_PQ_K, M * ds };
            add_side("pq_cb", GGML_TYPE_I8, 2, ne_cb, pack.cb8.data(), pack.cb8.size());
            const int64_t ne_inv[1] = { M * ds };
            add_side("pq_inv", GGML_TYPE_F32, 1, ne_inv, pack.inv.data(),
                     pack.inv.size() * sizeof(float));
            const int64_t ne_idx[2] = { t->ne[0], M };
            add_side("pq_idx", GGML_TYPE_I8, 2, ne_idx, pack.idx.data(), pack.idx.size());
        }
        n_pq++;
        fprintf(stderr, "%s: PQ packed '%s' (mode=%s ds=%d rel_mse=%.3e)\n", __func__, name,
                mode == 0 ? "S1" : "S2", ds, pack_rel_mse(t_tr, pack));
    }

    fprintf(stderr, "%s: %d tensors PQ-packed, writing %s ...\n", __func__, n_pq, out_path);
    if (!gguf_write_to_file(ctx_out, argv[2], false)) {
        fprintf(stderr, "%s: failed to write %s\n", __func__, argv[2]);
        return 1;
    }
    fprintf(stderr, "%s: done\n", __func__);
    return 0;
}
