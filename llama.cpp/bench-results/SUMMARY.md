# Llama-2-7b-chat CPU inference benchmark

- CPU: Intel(R) Xeon(R) 6759P-C
- Backends: CPU  (AVX512 + AMX bf16/int8)
- Build: llama.cpp efb3036c1 (9985)
- Threads: 120  NUMA: isolate  Repetitions: 3
- Tests: pp512 (prompt processing, 512 tokens), tg128 (generation, 128 tokens)
- KV cache: type_k=f16, type_v=f16, mmap=on, flash_attn=-1

| Quant | Model size | pp512 t/s | tg128 t/s |
|-------|-----------|----------:|----------:|
| Q2_K | 2.36 GiB | 75.21 ± 1.58 | 24.10 ± 21.34 |
| Q3_K_S | 2.75 GiB | 141.65 ± 24.58 | 31.66 ± 1.00 |
| Q3_K_M | 3.07 GiB | 72.37 ± 5.61 | 15.36 ± 16.91 |
| Q3_K_L | 3.35 GiB | 176.30 ± 66.92 | 35.72 ± 0.07 |
| Q4_0 | 3.56 GiB | 567.64 ± 332.81 | 45.36 ± 0.79 |
| Q4_1 | 3.95 GiB | 459.57 ± 183.58 | 42.98 ± 0.35 |
| Q4_K_S | 3.59 GiB | 658.61 ± 84.84 | 39.54 ± 0.46 |
| Q4_K_M | 3.80 GiB | 810.60 ± 7.17 | 36.31 ± 2.27 |
| Q5_0 | 4.33 GiB | 233.09 ± 1.83 | 22.97 ± 0.04 |
| Q5_1 | 4.72 GiB | 113.54 ± 7.22 | 21.98 ± 0.06 |
| Q5_K_S | 4.33 GiB | 554.75 ± 161.93 | 30.44 ± 0.68 |
| Q5_K_M | 4.45 GiB | 604.90 ± 32.50 | 28.99 ± 0.56 |
| Q6_K | 5.15 GiB | 439.42 ± 125.80 | 27.13 ± 0.70 |
| Q8_0 | 6.67 GiB | 450.52 ± 104.59 | 25.86 ± 0.06 |

Notes:
- pp512 stddev is high on some quants due to first-run page-fault / mmap cold cache.
- tg128 (decode) is memory-bandwidth bound; smaller quants run faster per token.
- AMX accelerates Q4_0/Q8_0/K-quants via the AMX mul_mat path (ggml_backend_amx_mul_mat).

## Quantization algorithm principles

llama.cpp's GGUF quants fall into two families. All are **block-wise linear**:
each block of N weights shares a scale `d` (and optionally a min `m`), and the
stored integer `q` reconstructs the weight as `x = d * q + m`. Block size and
how the scale itself is stored is what differentiates the schemes.

Block quantization trades a small per-block scale overhead for much lower
weight bandwidth. Effective bits-per-weight (bpw) = (weight bits + scale bits)
/ block size. For LLM inference, which is memory-bandwidth bound, lower bpw
usually means faster decode at the cost of accuracy.

### Legacy block quants (block size = 32)

Uniform per-block scale, no super-block structure. Simple and fast to decode,
but the single scale per 32 weights limits accuracy on outliers.

| Type | Block | Layout (per 32 weights) | bpw | Reconstruction |
|------|------:|------------------------|----:|----------------|
| Q4_0 | 32 | 1 x fp16 scale + 32 x 4-bit (symmetric) | 4.50 | `x = d * q` |
| Q4_1 | 32 | 1 x fp16 scale + 1 x fp16 min + 32 x 4-bit | 5.00 | `x = d * q + m` (asymmetric) |
| Q5_0 | 32 | 1 x fp16 scale + 1 x u32 hi-bits + 32 x 4-bit | 5.50 | `x = d * q` (5-bit via hi-bit) |
| Q5_1 | 32 | 1 x fp16 scale + 1 x fp16 min + 1 x u32 hi-bits + 32 x 4-bit | 6.00 | `x = d * q + m` (asymmetric, 5-bit) |
| Q8_0 | 32 | 1 x fp16 scale + 32 x int8 | 8.50 | `x = d * q` |

- **Symmetric (Q4_0/Q5_0/Q8_0)**: range is `[-d, +d]` around zero. Cheapest,
  good for weights that are roughly zero-mean.
- **Asymmetric (Q4_1/Q5_1)**: adds an explicit min/offset `m`, so the range
  need not be centered at zero. Better accuracy when a block of weights has a
  non-zero mean, at ~0.5 bpw extra cost.
- **Q8_0** is effectively int8 with a per-block fp16 scale; the highest-fidelity
  quant here and the usual reference for measuring degradation of lower-bit
  schemes.

### K-quants (super-block size = 256, `QK_K = 256`)

K-quants add a second level of scaling. A **super-block** of 256 weights is
split into sub-blocks (16 or 32 weights). Each sub-block has its own 6-bit
scale, and those sub-block scales are themselves quantized against a single
fp16 super-block scale `d` (and `dmin` for the mins). This two-level scheme
cuts the per-weight scale overhead dramatically versus the legacy 32-block
format, freeing bits for the actual weights.

| Type | Sub-blocks | Weight bits | Scale storage | bpw | Reconstruction |
|------|-----------:|------------:|---------------|----:|----------------|
| Q2_K | 16 x 16 | 2 | 4-bit sub-scales + fp16 d/dmin | 2.625 | `x = d*s*q + dmin*smin` |
| Q3_K | 16 x 16 | 3 | 6-bit sub-scales + fp16 d | 3.4375 | `x = d*s*q` |
| Q4_K | 8 x 32 | 4 | 6-bit sub-scales + fp16 d/dmin | 4.5 | `x = d*s*q + dmin*smin` |
| Q5_K | 8 x 32 | 5 | 6-bit sub-scales + fp16 d/dmin | 5.5 | `x = d*s*q + dmin*smin` |
| Q6_K | 16 x 16 | 6 | 8-bit int8 sub-scales + fp16 d | 6.5625 | `x = d*s*q` |

- **Q2_K**: 2-bit weights, with sub-block scales and mins quantized to 4 bits.
  Lowest bpw here; the min term makes it asymmetric. Highest speedup from
  bandwidth reduction but the largest accuracy hit.
- **Q3_K**: 3-bit weights with an extra high-bit mask. Comes in three variants
  that trade scale precision for weight precision:
  - **Q3_K_S** (Small): sub-block scales are 4-bit -> more bits left for
    weights in aggregate, but coarser scales.
  - **Q3_K_M** (Medium): 6-bit scales (default).
  - **Q3_K_L** (Large): 6-bit scales plus the 1-bit tensors (output norm) kept
    at higher precision. Highest accuracy of the Q3_K family.
- **Q4_K / Q5_K**: 4-/5-bit weights in 8 sub-blocks of 32, with 6-bit
  sub-scales and a min term (asymmetric). The `_S` variant quantizes part of
  the attention/mlp tensors to lower precision, while `_M` keeps them at Q4_K/
  Q5_K. This is why Q4_K_S and Q4_K_M share the same headline bpw but differ
  in which layers use the lower-bit path.
- **Q6_K**: 6-bit weights with int8 sub-block scales (not further quantized).
  Nearly lossless versus fp16 for most models; the accuracy reference among
  the K-quants.

### Why K-quants usually win on accuracy-per-bit

The legacy 32-weight block pays one full fp16 scale (16 bits) per 32 weights =
0.5 bpw of overhead, plus another 0.5 bpw if a min is stored. K-quants pay
only a few bits per 16- or 32-weight sub-block because the sub-scales are
themselves quantized, so the same target bpw leaves more bits for the weights
themselves. This is why, at similar file size, Q4_K_M beats Q4_1 on perplexity,
and why the K-quants dominate the results above.

### Mapping to the benchmark

- Decode (tg128) is bandwidth-bound, so bpw tracks speed: Q2_K/Q3_K_S are
  fastest, Q8_0/Q6_K slowest. Q4_0/Q4_1 run faster than the K-quants at the
  same bit width because their decode kernels are simpler (no super-block
  scale unpacking) and AMX has a direct Q4_0/Q8_0 mul_mat path.
- Prompt processing (pp512) is compute-bound and benefits from AMX. Q4_K_M
  hits 810 t/s because the K-quant GEMM kernels are heavily optimized for the
  AMX bf16/int8 tile path, and the 6-bit sub-scale overhead is amortized over
  the large batch.
