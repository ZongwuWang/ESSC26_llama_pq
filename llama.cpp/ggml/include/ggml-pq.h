// PQ (product quantization) decode-time GEMV for llama.cpp CPU backend.
//
// Linear weights are PQ-compressed offline/at-load into per-subspace
// codebooks + 1 byte/weight indices (d_sub=2..8, K=256). During decode
// (n_tokens==1) ggml_compute_forward_mul_mat intercepts registered tensors
// and runs the PQ GEMV kernels ported from pq_cpu/bench_pq.cpp. Prefill and
// unregistered tensors keep the original path (original weights stay valid).
//
// Modes:
//   S1 (0): partition along the INPUT dim. y[j] = sum_i dt_i[idx_i[j]],
//           dt_i[k] = sum_d x[i*ds+d] * c[i][k][d]. Codebook stored fp16
//           (SoA, [(i*ds+d)*K+k]); dt tables are rebuilt per GEMV and
//           int8-quantized on the fly (vpermi2b byte-LUT).
//   S2 (1): partition along the OUTPUT dim. y[i*ds+s] = sum_j x[j]*c_s(idx_ij).
//           Codebook stored int8 with per-(i,side) dequant scale; x scaled
//           by 1/128 to keep the fp16 accumulator in range.
#ifndef GGML_PQ_GEMV_H
#define GGML_PQ_GEMV_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_PQ_MAX_DS 8
#define GGML_PQ_K      256

// Same-input GEMV fusion: consecutive MUL_MAT nodes sharing one input row
// (wq/wk/wv after an rms_norm, ffn_gate/ffn_up) are computed in a single
// barrier-coordinated pass, cutting per-node sync overhead and tripling the
// contiguous weight-stream length. The caller drives it with three hooks:
//   - ggml_pq_mul_mat_fused at each decode-shaped MUL_MAT (append/defer/compute)
//   - ggml_pq_node_boundary at every other node (flush before dependents read)
//   - ggml_pq_graph_end after the node loop (flush a graph-final group)
// All threads of the pool must call the hooks for every node, in the same
// order (group state is per-thread and kept in lockstep by node barriers).
#define GGML_PQ_MM_NONE     0  // not a PQ tensor: run the normal path
#define GGML_PQ_MM_DONE     1  // output written
#define GGML_PQ_MM_DEFERRED 2  // joined the pending group; output written on flush
int  ggml_pq_mul_mat_fused(const char * name, const void * key, const void * x,
                           int x_type, float * dst, int64_t n_in, int64_t n_out,
                           int ith, int nth, void * threadpool);
void ggml_pq_node_boundary(const void * node /* struct ggml_tensor* */,
                           int ith, int nth, void * threadpool);
void ggml_pq_graph_end(int ith, int nth, void * threadpool);

// drop all registered tensors (call before re-registration)
void ggml_pq_reset(void);
// flip the global decode interception switch
void ggml_pq_set_enabled(bool v);
bool ggml_pq_enabled(void);

// mode 0 = S1 (cb_f32: [(i*K+k)*ds+d], converted to fp16 SoA internally)
// mode 1 = S2 (cb8: [(i*ds+s)*K+k] int8, inv: [i*ds+s] dequant scale;
//              inv is pre-multiplied by 128 here to pair with x/128)
bool ggml_pq_register(const char * name, int mode, int ds,
                      const float * cb_f32,
                      const int8_t * cb8, const float * inv,
                      const uint8_t * idx,
                      int64_t n_in, int64_t n_out);

// pre-packed registration (as persisted in GGUF):
//   S1: cb is fp16 SoA [(i*ds+d)*K + k], no scales (runtime dt quantization)
//   S2: cb is int8 [(i*ds+s)*K + k], inv [i*ds+s] in FLOAT units (scaled by
//       128 internally here, pairing with the x/128 kernel input)
bool ggml_pq_register_raw(const char * name, int mode, int ds,
                          const void * cb,
                          const float * inv,
                          const uint8_t * idx,
                          int64_t n_in, int64_t n_out);

bool ggml_pq_register_raw_scaled(const char * name, int ds,
                                 const void * cb, const uint8_t * idx,
                                 const float * row_scale,
                                 int64_t n_in, int64_t n_out);

// try to run y[n_out] = W_pq * x[n_in] for one decode row.
// x_type: GGML type of x (F32 or F16 supported).
// threadpool: ggml_threadpool* for ggml_barrier (S1 two-phase scheme).
// returns false if `name` is not registered (caller falls back).
bool ggml_pq_mul_mat_vec(const char * name, const void * x, int x_type,
                         float * dst, int64_t n_in, int64_t n_out,
                         int ith, int nth, void * threadpool);

// node-timing aggregation helpers (used by the GGML_NODE_TIMING instrumentation)
void ggml_pq_timing_note_mulmat(const char * src_name, uint64_t busy_us);
void ggml_pq_timing_dump_mulmat(void);

// Graph-level fusion group: register `name` (e.g. "blk.0.attn_qkv", the fused
// weight of a graph-level QKV mul_mat) as ONE PQ GEMV that runs the already
// registered S1 `members` back-to-back in a single barrier pass, writing each
// member's output to dst + row_off[m]. Members must be S1 with equal n_in and
// equal ds; their individual nodes no longer exist in the graph.
bool ggml_pq_register_group(const char * name, int64_t n_in, int64_t n_out_total,
                            const char * const * members, int n_members,
                            const int64_t * row_offs);

#ifdef __cplusplus
}
#endif

#endif // GGML_PQ_GEMV_H
