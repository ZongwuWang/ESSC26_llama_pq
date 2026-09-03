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

[Why EdgePQ](#why-edgepq) · [Results](#main-results) · [Demo](#demo) · [Requirements](#requirements) · [Quick Start](#quick-start) · [Reproduction](#step-by-step-reproduction) · [Commands](#command-reference)

</div>

## Why EdgePQ

- **Two-bit CPU-native decoding.** Learned non-uniform codebooks reduce the
  weight stream while keeping the token-generation path on the CPU.
- **Dual-mode PQ GEMV runtime.** The implementation supports input-partitioned
  S1 and output-partitioned S2 layouts for different matrix shapes.
- **Reproducible artifact.** A locked `uv` environment and Make targets cover
  input preparation, correctness checks, throughput, perplexity, and plots.

## Main Results

The released artifact uses the following reference results with 60 CPU threads
on the evaluation platform described in the paper:

| Model | TG128 (tok/s) | TG512 (tok/s) | WikiText-2 PPL |
|---|---:|---:|---:|
| F16 | 25.24 ± 0.38 | 24.10 ± 0.77 | 6.09 |
| Q2_K | 38.11 ± 0.89 | 34.24 ± 0.37 | 7.75 |
| **EdgePQ** | **56.50 ± 0.64** | **53.04 ± 0.58** | **6.34** |

EdgePQ is 2.24×/2.20× faster than F16 and 48.3%/54.9% faster than Q2_K at
TG128/TG512. Its PPL is 18.2% lower than Q2_K and within 4.1% of F16. The
unrounded values are 6.0944 (F16), 7.7528 (Q2_K), and 6.3443 (EdgePQ); the
table reports two-decimal values.

EdgePQ PPL is measured by reconstructing the trained PQ checkpoint into an
FP16 Transformers model and evaluating complete, non-overlapping 4096-token
WikiText-2 windows. Throughput loads the PQ side tensors embedded in
`base-pq-4c8b.gguf`. The released 4c8b checkpoint stores all 224 trained linear
states with input-axis partitioning (`axis=1`), so this artifact uses the S1
AVX-512 LUT kernel with `d=4`; the generic runtime additionally supports S2.

> [!IMPORTANT]
> The table contains the released artifact's reference values. Every
> reproduction writes its own measurements to `output/throughput.csv` and
> `output/perplexity.csv`.
> Absolute CPU throughput can vary with CPU load, NUMA placement, page-cache
> state, and background contention; compare runs under the same conditions.

## Demo

The online demonstration shows the EdgePQ artifact running from the public
repository:

https://github.com/user-attachments/assets/fdb0e12c-c5cb-42de-a32a-d360a3e8d43d

A separate downloadable artifact walkthrough is retained in
[`video_demo/demo.mp4`](video_demo/demo.mp4).

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
locked `.venv`, downloads only missing or stale inputs, validates them, builds or reuses
the runtime, runs `pq-selftest`, and loads all three models in an eight-token
smoke test. It reuses result CSVs only when their cache key matches the pinned
models and evaluation protocol; otherwise it runs the complete benchmark and
PPL evaluation. It then prints the result table and renders both charts.

The first run downloads approximately 60 GB and can take one to three hours.
Later runs reuse the environment, inputs, build, and existing result CSVs.

### Interactive Generation

```bash
make chat MODEL=pq
make chat MODEL=16
make chat MODEL=k2
```

`MODEL=pq`, `MODEL=16`, and `MODEL=k2` select EdgePQ, FP16, and Q2_K,
respectively; omitting `MODEL` defaults to EdgePQ. Each command prepares only
the selected model and reuses it when already present. The FP16 and Q2_K modes
disable PQ execution, while the EdgePQ mode enables its embedded PQ side
tensors. EdgePQ defaults to deterministic greedy decoding (`temperature=0`);
FP16 and Q2_K retain `temperature=0.7`. Override either behavior with
`CHAT_TEMPERATURE`. The runner inspects the GGUF metadata automatically. A model containing
`tokenizer.chat_template` uses llama.cpp's chat-template API and retains the
standard system/user/assistant message history. A model without that metadata
uses completion mode: each entered prompt is an independent continuation and
no synthetic chat format is added.
The selected mode and Prompt/Generation tokens per second are printed in the
terminal. Use `/clear` to clear chat history and `/exit` to stop.

Override the automatic decision or select another compatible GGUF when needed:

```bash
make chat GGML_CUDA=OFF CHAT_MODE=completion
make chat GGML_CUDA=OFF CHAT_MODEL=/data/chat-edgepq.gguf CHAT_MODE=auto
```

`CHAT_MODE` accepts `auto`, `chat`, or `completion`. Forced chat mode fails
clearly if the GGUF has no embedded chat template. Other useful overrides are
`CHAT_THREADS`, `CHAT_CPU_RANGE`, `CHAT_CONTEXT`, `CHAT_MAX_TOKENS`, and
`CHAT_SYSTEM`. For example, this limits each response to 128 generated tokens
and restores stochastic sampling at temperature 0.7:

```bash
make chat MODEL=pq CHAT_MAX_TOKENS=128 CHAT_TEMPERATURE=0.7
```

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

## Command Reference

`make help` prints the same public target list from the Makefile.

| Command | Purpose | Main output |
|---|---|---|
| `make help` | List the public artifact targets | Terminal command list |
| `make env` | Create or update `.venv` from `uv.lock` | `.venv/` |
| `make prepare-inputs` | Download only missing models, checkpoint, tokenizer, and WikiText-2 | `models/`, `datasets/` |
| `make check-inputs` | Validate every required input and all 224 PQ states | Terminal PASS/failure |
| `make llama-build` | Build llama.cpp, PPL, quantizer, and PQ tools | `build/bin/` |
| `make selftest` | Compare registered PQ execution with the reference path | Terminal PASS/failure |
| `make smoke` | Load F16, Q2_K, and EdgePQ and generate eight tokens each | Terminal validation |
| `make chat MODEL=pq\|16\|k2` | Select EdgePQ, FP16, or Q2_K and start interactive generation | Terminal conversation and Token/s |
| `make demo` | Run revision-aware end-to-end reproduction, including smoke and plots | CSVs, PNGs, terminal summary |
| `make benchmark` | Measure F16, Q2_K, and EdgePQ decode throughput | `output/throughput.csv` |
| `make ppl` | Evaluate F16 and Q2_K perplexity | `output/perplexity.csv` |
| `make pq-ppl` | Evaluate EdgePQ reconstruction perplexity | Appends to `output/perplexity.csv` |
| `make all` | Rerun the complete paper evaluation; does not render plots | Both result CSVs |
| `make plot` | Print and render both result CSVs | Two PNG charts |

The distinction between the two aggregate targets is intentional: `make demo`
may reuse CSVs matching the current model/protocol cache key and always runs
the smoke test and plotting stage;
`make all` always reruns the measurements and leaves plotting as a separate
`make plot` step.

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
|   |-- tools/pq-chat/    # auto-detected Chat/completion frontend
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
