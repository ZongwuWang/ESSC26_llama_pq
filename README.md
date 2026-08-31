# EdgePQ

EdgePQ is a CPU-native 2-bit product-quantized LLM decoding system built on a
vendored, PQ-extended version of [llama.cpp](https://github.com/ggml-org/llama.cpp).
It uses input-partitioned (S1) and output-partitioned (S2) PQ layouts to execute
Llama-2-7B decoding on many-core x86 CPUs.

## Main Results

The paper reports the following results with 60 CPU threads:

| Model | TG128 | TG512 | WikiText-2 PPL |
|---|---:|---:|---:|
| F16 | 18.80 +/- 0.79 | 20.55 +/- 0.34 | 6.12 |
| Q2_K | 48.07 +/- 1.35 | 47.80 +/- 0.79 | 7.74 |
| EdgePQ | **55.35 +/- 0.64** | **50.73 +/- 1.87** | **6.33** |

The EdgePQ PPL value is measured by reconstructing the trained PQ checkpoint
into an FP16 Transformers model. Throughput uses the PQ side tensors embedded
in `base-pq-4c8b.gguf`.

**Video demo:** see [`video_demo/`](video_demo/).

## Requirements

- Linux x86-64
- AVX-512 FP16, VBMI, and VNNI for the optimized PQ path
- GCC/G++, CMake, Ninja, and `uv`
- 64 GB RAM or more
- NVIDIA GPU with at least 32 GB memory for the full PPL run
- About 60 GB free disk space

The CPU throughput path does not require a GPU. CUDA is enabled in the default
build because `make ppl` and `make pq-ppl` use GPU evaluation.

## Repository Layout

```text
ESSC26_llama_pq/
|-- pyproject.toml        # uv environment specification
|-- uv.lock               # locked Python dependencies
|-- Makefile              # all artifact commands
|-- llama_pq.cpp          # FP16 vs Q2_K vs EdgePQ throughput
|-- ppl_gguf_compare.py   # dataset and PPL evaluation
|-- plot_results.py       # throughput and PPL charts
|-- llama.cpp/            # vendored, PQ-extended llama.cpp
|   |-- tools/pq-convert/ # checkpoint exporter
|   `-- tools/pq-selftest/# correctness self-test
|-- video_demo/           # artifact walkthrough
`-- output/               # generated CSVs and charts (not tracked)
```

## Prepare Inputs

The public model repository
[`ZongwuWang/EdgePQ-4c8b`](https://huggingface.co/ZongwuWang/EdgePQ-4c8b)
contains:

- `base-pq-4c8b.gguf`, which carries the deployable EdgePQ side tensors (the
  paper's FP16 baseline remains the separate `Llama-2-7b-chat-hf-f16.gguf`);
- `best-formal-hard/non_pq_state.pt`;
- all 224 files under `best-formal-hard/pq_states/`;
- `tokenizer.json`, `tokenizer.model`, and `tokenizer_config.json`.

The prepared inputs are loaded directly: `Llama-2-7b-chat-hf-f16.gguf` for FP16,
`Llama-2-7b-chat-hf-Q2_K.gguf` for Q2_K, and `base-pq-4c8b.gguf` for PQ-4c8b.
The artifact does not rename or regenerate these files.

Create the Python environment and prepare the default inputs:

```bash
make env
make prepare-inputs
```

This downloads the public EdgePQ GGUF/checkpoint/tokenizer, the F16 and Q2_K GGUFs from
[`second-state/Llama-2-7B-Chat-GGUF`](https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF),
and WikiText-2 directly from Hugging Face.

Default paths:

```text
models/base-pq-4c8b.gguf
models/Llama-2-7b-chat-hf-f16.gguf
models/Llama-2-7b-chat-hf-Q2_K.gguf
models/best-formal-hard/
models/tokenizer/
datasets/wikitext-2-raw-v1/
```

All paths are Make variables and can be overridden:

```bash
make all \
  PQ_MODEL=/data/base-pq-4c8b.gguf \
  FP16_MODEL=/data/Llama-2-7b-chat-hf-f16.gguf \
  Q2_MODEL=/data/Llama-2-7b-chat-hf-Q2_K.gguf \
  PQ_CHECKPOINT=/data/best-formal-hard \
  PPL_DATASET=/data/wikitext-2-raw-v1 \
  TOKENIZER=/data/llama2-tokenizer \
  CUDA_DEVICE=0
```

## Build

```bash
make env
make llama-build
```

The build includes `llama-perplexity`, `llama-quantize`, `pq-selftest`, and
`llama-pq-convert`, followed by the root `llama_pq` runner when needed.

## Quick Check

Validate all external inputs:

```bash
make check-inputs
```

Run the standalone correctness test and a short three-model decode test:

```bash
make selftest
make smoke
```

The smoke target runs eight generated tokens with one measured repetition and
does not write a result file.

## Reproduce the Paper

Run the complete evaluation:

```bash
make all
```

Render the CSV results:

```bash
make plot
```

Individual stages are also available:

```bash
make benchmark
make ppl
make pq-ppl
```

The complete PPL evaluation can take one to three hours.

The paper settings are fixed in the Makefile: F16/Q2_K use context 3072 and
stride 2048; PQ reconstruction uses context/stride 4096.

## Output Files

```text
output/throughput.csv
output/perplexity.csv
output/throughput.png
output/ppl.png
```

No persistent run logs are generated. Command output is printed to the terminal,
and any failed stage returns a non-zero exit status.

## Clean Up

Remove only the root runner:

```bash
make clean
```

Remove the build, virtual environment, and generated output:

```bash
make distclean
```

Model and dataset inputs are never removed by these targets.

## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [T-MAC](https://github.com/microsoft/T-MAC)
- Product quantization work cited in the accompanying paper

## License

This repository is released under the terms in [`LICENSE`](LICENSE). The
Llama-2 model and derived artifacts remain subject to their applicable licenses.
