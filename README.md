<div align="center">

<h1>EdgePQ</h1>

<p><strong>CPU-native two-bit product-quantized LLM decoding for many-core x86.</strong></p>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform: Linux x86-64](https://img.shields.io/badge/platform-Linux%20x86--64-2f363d)
![CPU: AVX-512](https://img.shields.io/badge/CPU-AVX--512-7b61ff)
[![Model: Llama-2-7B](https://img.shields.io/badge/model-Llama--2--7B-f4a261)](https://huggingface.co/ZongwuWang/EdgePQ-4c8b)

EdgePQ co-designs product quantization and SIMD lookup-table execution to run
Llama-2-7B decoding directly on commodity CPUs, with no GPU in the decoding
critical path.

[Why EdgePQ](#why-edgepq) · [Results](#main-results) · [Demo](#demo) · [Requirements](#requirements) · [Quick Start](#quick-start) · [Reproduction](#step-by-step-reproduction)

</div>

## Why EdgePQ

- **Two-bit CPU-native decoding.** Learned non-uniform codebooks reduce the
  weight stream while keeping the token-generation path on the CPU.
- **Dual-mode PQ GEMV.** Input-partitioned S1 and output-partitioned S2 layouts
  target different matrix shapes and are selected per layer.
- **Reproducible artifact.** A locked `uv` environment and Make targets cover
  input preparation, correctness checks, throughput, perplexity, and plots.

## Main Results

The paper reports the following results with 60 CPU threads on the evaluation
platform described in the paper:

| Model | TG128 (tok/s) | TG512 (tok/s) | WikiText-2 PPL |
|---|---:|---:|---:|
| F16 | 25.24 ± 0.38 | 24.10 ± 0.77 | 6.09 |
| Q2_K | 38.11 ± 0.89 | 34.24 ± 0.37 | 7.75 |
| **EdgePQ** | **56.50 ± 0.64** | **53.04 ± 0.58** | **6.33** |

EdgePQ is 2.24×/2.20× faster than F16 and 48.3%/54.9% faster than Q2_K at
TG128/TG512. Its PPL is 18.3% lower than Q2_K and within 3.9% of F16. The
unrounded reproduced PPL values are 6.0944 (F16), 7.7528 (Q2_K), and 6.3286
(EdgePQ); the table shows the two-decimal values used in the paper.

EdgePQ PPL is measured by reconstructing the trained PQ checkpoint into an
FP16 Transformers model and evaluating complete, non-overlapping 4096-token
WikiText-2 windows. Throughput loads the PQ side tensors embedded in
`base-pq-4c8b.gguf`.

> [!IMPORTANT]
> The table contains the paper reference values. Every reproduction writes its
> own measurements to `output/throughput.csv` and `output/perplexity.csv`.
> Absolute CPU throughput can vary with CPU load, NUMA placement, page-cache
> state, and background contention; compare runs under the same conditions.

## Demo

The ten-minute walkthrough covers environment and input preparation,
correctness checks, the three-model smoke test, evaluation, and result
inspection. [Watch or download the video](video_demo/demo.mp4).

## Requirements

| Workflow | Requirements |
|---|---|
| CPU decode, self-test, and throughput | Linux x86-64; AVX-512 FP16, VBMI, and VNNI; GCC/G++; CMake; Ninja; `uv`; 64 GB RAM or more |
| Full PPL reproduction | All CPU requirements plus an NVIDIA GPU with at least 32 GB memory and a compatible CUDA toolkit |
| Complete artifact download | About 60 GB free disk space |

The optimized decoding path itself does not require a GPU. The default build
uses `GGML_CUDA=ON` because the complete PPL workflow uses CUDA. For a CPU-only
build and decode smoke test, use:

```bash
make llama-build GGML_CUDA=OFF
make selftest GGML_CUDA=OFF
make smoke GGML_CUDA=OFF
```

Do not run `make ppl` or `make pq-ppl` in the CPU-only workflow.

## Quick Start

```bash
git clone https://github.com/ZongwuWang/ESSC26_llama_pq.git
cd ESSC26_llama_pq
make demo
```

`make demo` is the one-command artifact entry point. It creates or reuses the
locked `.venv`, downloads only missing inputs, validates them, builds or reuses
the runtime, runs `pq-selftest`, and loads all three models in an eight-token
smoke test. If both result CSVs are non-empty, it reuses them; otherwise it runs
the complete benchmark and PPL evaluation. It then prints the result table and
renders both charts.

The first run downloads approximately 60 GB and can take one to three hours.
Later runs reuse the environment, inputs, build, and existing result CSVs.

## Step-by-Step Reproduction

### 1. Environment and Inputs

```bash
make env
make prepare-inputs
make check-inputs
```

`make env` installs the locked Python environment. `make prepare-inputs`
downloads only missing files, and `make check-inputs` verifies the three GGUFs,
all 224 PQ states, tokenizer files, and WikiText-2.

| Input | Default path | Public source |
|---|---|---|
| EdgePQ | `models/base-pq-4c8b.gguf` | [`ZongwuWang/EdgePQ-4c8b`](https://huggingface.co/ZongwuWang/EdgePQ-4c8b) |
| F16 | `models/Llama-2-7b-chat-hf-f16.gguf` | [`second-state/Llama-2-7B-Chat-GGUF`](https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF) |
| Q2_K | `models/Llama-2-7b-chat-hf.Q2_K.gguf` | [`ZongwuWang/EdgePQ-4c8b`](https://huggingface.co/ZongwuWang/EdgePQ-4c8b) |
| Checkpoint and tokenizer | `models/best-formal-hard/`, `models/tokenizer/` | [`ZongwuWang/EdgePQ-4c8b`](https://huggingface.co/ZongwuWang/EdgePQ-4c8b) |
| WikiText-2 | `datasets/wikitext-2-raw-v1/` | [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) |

Model revisions are pinned in the Makefile, and the WikiText-2 revision is
pinned in `ppl_gguf_compare.py`. The artifact loads these files directly and
does not rename or regenerate the baseline GGUFs. Repeated runs reuse completed
files.

### 2. Build and Validate

```bash
make llama-build
make selftest
make smoke
```

The build includes `llama-perplexity`, `llama-quantize`, `pq-selftest`, and
`llama-pq-convert`. `make selftest` compares registered PQ execution with the
reference path; `make smoke` loads F16, Q2_K, and EdgePQ and generates eight
tokens with each model.

### 3. Run Measurements

```bash
make benchmark
make ppl
make pq-ppl
```

| Command | Purpose | Main output |
|---|---|---|
| `make benchmark` | Measure F16, Q2_K, and EdgePQ decode throughput | `output/throughput.csv` |
| `make ppl` | Evaluate F16 and Q2_K perplexity | `output/perplexity.csv` |
| `make pq-ppl` | Evaluate reconstructed EdgePQ perplexity | Appends to `output/perplexity.csv` |

The complete PPL evaluation can take one to three hours. The default Make
variables match the paper protocol: F16/Q2_K use context 3072 and stride 2048;
PQ reconstruction uses context and stride 4096. These defaults can be
overridden when running controlled experiments.

### 4. Render Results

```bash
make plot
```

`make plot` prints the measured values and renders the throughput and PPL
charts. To force every measurement to rerun, use `make all` followed by
`make plot`.

## Output Files

| File | Contents |
|---|---|
| `output/throughput.csv` | Per-model TG128/TG512 throughput measurements |
| `output/perplexity.csv` | F16, Q2_K, and EdgePQ perplexity measurements |
| `output/throughput.png` | Throughput chart generated by `make plot` |
| `output/ppl.png` | Perplexity chart generated by `make plot` |

No persistent run logs are generated. Command output is printed to the
terminal, and a failed stage returns a non-zero exit status.

## Advanced Usage

<details>
<summary><strong>Reuse a persistent input cache</strong></summary>

Models and datasets can be shared across checkouts:

```bash
CACHE_ROOT=/data/edgepq-artifact-cache

make prepare-inputs \
  MODEL_DIR="$CACHE_ROOT/models" \
  DATASET_DIR="$CACHE_ROOT/datasets"

make demo \
  MODEL_DIR="$CACHE_ROOT/models" \
  DATASET_DIR="$CACHE_ROOT/datasets"
```

Hugging Face downloads, the `uv` package cache, and the Ninja build are reused
when their completed files already exist.

</details>

<details>
<summary><strong>Use custom input paths</strong></summary>

All input paths and runtime choices are Make variables:

```bash
make demo \
  PQ_MODEL=/data/base-pq-4c8b.gguf \
  FP16_MODEL=/data/Llama-2-7b-chat-hf-f16.gguf \
  Q2_MODEL=/data/Llama-2-7b-chat-hf.Q2_K.gguf \
  PQ_CHECKPOINT=/data/best-formal-hard \
  PPL_DATASET=/data/wikitext-2-raw-v1 \
  TOKENIZER=/data/llama2-tokenizer \
  CUDA_DEVICE=0
```

</details>

<details>
<summary><strong>Repository layout</strong></summary>

```text
ESSC26_llama_pq/
|-- pyproject.toml        # uv environment specification
|-- uv.lock               # locked Python dependencies
|-- Makefile              # artifact commands and defaults
|-- llama_pq.cpp          # F16 vs Q2_K vs EdgePQ throughput
|-- ppl_gguf_compare.py   # dataset preparation and PPL evaluation
|-- plot_results.py       # throughput and PPL charts
|-- llama.cpp/            # vendored, PQ-extended llama.cpp
|   |-- tools/pq-convert/ # checkpoint exporter
|   `-- tools/pq-selftest/# correctness self-test
|-- video_demo/           # recorded artifact walkthrough
`-- output/               # generated CSVs and charts, not tracked
```

</details>

## License

This repository is released under the terms in [`LICENSE`](LICENSE). The
Llama-2 model and derived artifacts remain subject to their applicable
licenses.
