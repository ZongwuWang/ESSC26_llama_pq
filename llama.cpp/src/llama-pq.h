// Load-time PQ quantization of linear-layer weights for decode-time GEMV,
// plus (de)serialization of PQ packs to side tensors inside a GGUF file.
#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

struct ggml_tensor;
struct llama_model_loader;

// reset registry and latch ds/mode from model params
void llama_pq_begin(const llama_model_params & params);

// arm the decode interception (call after all tensors are registered)
void llama_pq_finish(void);

// PQ pack for one 2-D weight tensor. Layouts match the ggml-cpu registry
// (ggml-pq.h) so packs can be registered directly or serialized.
struct llama_pq_pack {
    int     mode = 0;            // 0 = S1 (input-dim), 1 = S2 (output-dim)
    int     ds = 2;
    int64_t n_in = 0, n_out = 0;
    // S1: [(i*K+k)*ds+d] fp32 (K centroids per subspace, ds dims each)
    std::vector<float>   cbf;
    // S2: [(i*ds+s)*K+k] int8 + dequant scales [i*ds+s] (float units)
    std::vector<int8_t>  cb8;
    std::vector<float>   inv;
    // S1: [i*n_out + j]; S2: [i*n_in + j]
    std::vector<uint8_t> idx;
};

// Quantize `t` ((n_out, n_in) 2-D weight, any ggml type, contiguous) with a
// product-quantizer (per-subspace k-means, grid init + Lloyd iterations,
// following TFLOP's faiss.ProductQuantizer approach).
//
// mode 0 (S1): subspaces along the INPUT dim  (TFLOP axis=1; q/k/v/o, gate/up)
// mode 1 (S2): subspaces along the OUTPUT dim (TFLOP axis=0; down_proj)
//
// Returns false if the tensor is unsuitable (too small, dims not divisible
// by ds, non-contiguous).
bool llama_pq_build_pack(struct ggml_tensor * t, int mode, int ds,
                         llama_pq_pack & out);

// Quantize + register with the ggml-cpu PQ registry. Returns false if the
// tensor is unsuitable.
bool llama_pq_register_tensor(struct ggml_tensor * t, int mode, int ds);

// If GGUF contains pre-computed PQ side tensors for `base` (e.g.
// "blk.0.attn_q.weight.pq_meta/.pq_cb/.pq_idx"), load them and register.
// Returns false when the side tensors are absent/invalid (caller should
// fall back to llama_pq_register_tensor).
bool llama_pq_register_from_loader(struct llama_model_loader & ml,
                                   const char * base, int * ds_out = nullptr);

// GGUF side-tensor naming (shared with the pq-convert tool)
void llama_pq_side_name(char * buf, size_t cap, const char * base, const char * suffix);

// Graph-level QKV fusion: build a fused weight tensor named after
// "<wq->name with .attn_q.weight -> .attn_qkv>" whose rows are
// [wq; wk; wv] (row offsets returned in row_offs). Returns nullptr when the
// three tensors are not row-fusable (type/shape mismatch). The storage lives
// for the process lifetime (freed at the next model load).
struct ggml_tensor * llama_pq_build_qkv_weight(struct ggml_tensor * wq,
        struct ggml_tensor * wk, struct ggml_tensor * wv, int64_t * row_offs);

// Register the members' PQ side data (from GGUF or on-the-fly k-means) and a
// fused-group entry under `fused_name`. Returns false if any member failed
// (caller falls back to the separate per-tensor path).
bool llama_pq_register_qkv_group(struct llama_model_loader & ml,
        const char * fused_name, struct ggml_tensor * wq,
        struct ggml_tensor * wk, struct ggml_tensor * wv, int * ds_out);

// Generic forms used by the graph-level fusions (QKV, gate/up):
// build a fused weight of `n_ts` row-compatible tensors, renaming a suffix
// `name_from` of ts[0]'s name to `name_to`; register the members' side data
// plus a fused-group entry.
struct ggml_tensor * llama_pq_build_fused_weight(struct ggml_tensor ** ts,
        int n_ts, const char * name_from, const char * name_to, int64_t * row_offs);
bool llama_pq_register_fused_group(struct llama_model_loader & ml,
        const char * fused_name, struct ggml_tensor ** members, int n_members,
        const int64_t * row_offs, int * ds_out);
