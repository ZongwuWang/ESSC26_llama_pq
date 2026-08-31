# AE Usage Guide

This directory is a self-contained artifact-evaluation project for comparing
FP16, Q2_K, and PQ-4c8b Llama-2-7B decode performance. Run all commands below
from this directory.

## 1. Requirements

- A Linux x86-64 system with a C++17 compiler.
- CMake and Ninja.
- The CPU features required by the selected llama.cpp build. The PQ path is
  intended for AVX-512 FP16 and VBMI CPUs.
- Three compatible Llama-2-7B GGUF files:
  - an F16 model;
  - a standard llama.cpp Q2_K model;
  - the generated PQ-4c8b model.

The three GGUF model paths are provided as command-line arguments or Makefile
variables.

## 2. Expected repository layout

The project layout is:

```text
.
├── CMakeLists.txt
├── Makefile
├── README.md
├── llama_pq.cpp
└── llama.cpp/
    ├── CMakeLists.txt
    ├── include/
    ├── ggml/
    ├── common/
    └── src/
```

Model files must be available at the paths specified for the evaluation.

## 3. Build the bundled llama.cpp

The Makefile automatically configures and builds the llama.cpp source bundled
under `./llama.cpp`:

```bash
make llama-build
```

The build output is placed in `build`.

## 4. Build the AE program

### Makefile build

```bash
make
```

The executable is `./llama_pq`. The default Makefile target also builds the
bundled llama.cpp libraries when needed.

### CMake build

```bash
rm -rf build

env -i HOME="$HOME" \
  PATH=/usr/local/bin:/usr/bin:/bin \
  cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SRC="$PWD/llama.cpp" \
  -DLLAMA_BUILD="$PWD/build"

env -i HOME="$HOME" \
  PATH=/usr/local/bin:/usr/bin:/bin \
  ninja -C build
```

## 5. Run the complete AE comparison

The simplest reproducible command is:

```bash
make benchmark
```

This command:

- builds `./llama_pq` if needed;
- evaluates FP16, Q2_K, and PQ-4c8b;
- uses 60 CPU threads;
- uses distributed NUMA placement;
- performs one one-token warmup per model and generation length;
- runs TG128 and TG512;
- repeats each test three times;
- writes the CSV result to `essc-pq-comparison.csv`.

Inspect the result with:

```bash
cat essc-pq-comparison.csv
```

The expected CSV schema is:

```text
model,path,generation,tokens_per_second,stdev
F16,...,128,...,...
F16,...,512,...,...
Q2_K,...,128,...,...
Q2_K,...,512,...,...
PQ-4c8b,...,128,...,...
PQ-4c8b,...,512,...,...
```

## 6. Run with explicit parameters

Run the evaluator directly with explicit model paths:

```bash
GGML_PQ_STRIPE=1 \
OMP_DYNAMIC=FALSE \
OMP_PROC_BIND=spread \
OMP_PLACES=cores \
./llama_pq \
  --fp16 ../../models/Llama-2-7b-chat-hf.gguf \
  --q2 ../../models/Llama-2-7b-chat-hf-GGUF/Llama-2-7b-chat-hf.Q2_K.gguf \
  --pq ../../llama.cpp/base-pq-4c8b.gguf \
  --threads 60 \
  --repetitions 3 \
  --warmup 1 \
  --generations 128,512 \
  --context 2048 \
  --numa distribute \
  --output essc-pq-comparison.csv
```

Available options:

| Option | Default | Description |
|---|---:|---|
| `--fp16 PATH` | required | F16 GGUF model |
| `--q2 PATH` | required | standard Q2_K GGUF model |
| `--pq PATH` | required | PQ-4c8b GGUF model |
| `--threads N` | `60` | generation and batch thread count |
| `--repetitions N` | `3` | measured repetitions |
| `--warmup N` | `1` | one-token warmup repetitions |
| `--generations LIST` | `128,512` | comma-separated generation lengths |
| `--context N` | `2048` | minimum context size; actual size is at least generation length |
| `--numa MODE` | `distribute` | `distribute`, `isolate`, or `disabled` |
| `--output PATH` | stdout | CSV output path |

## 7. Override Makefile parameters

```bash
make benchmark \
  AE_THREADS=60 \
  AE_REPETITIONS=5 \
  AE_WARMUP=1 \
  AE_GENERATIONS=128,512 \
  AE_CONTEXT=2048 \
  AE_NUMA=distribute \
  AE_OUTPUT=my-ae.csv
```

Override model locations as follows:

```bash
make benchmark \
  FP16_MODEL=/data/models/model-f16.gguf \
  Q2_MODEL=/data/models/model.Q2_K.gguf \
  PQ_MODEL=/data/models/base-pq-4c8b.gguf
```

## 8. Correctness and runtime checks

Before performance testing, verify that the llama.cpp libraries are found:

```bash
ldd ./llama_pq | grep -E 'libllama|libggml|not found'
```

No `not found` line should be printed.

For a quick CLI check:

```bash
./llama_pq --help
```

## 9. Interpreting results

The reported value is generated tokens per second. It is measured only around
`llama_decode` calls and synchronization; model loading is excluded from the
throughput measurement. The F16, Q2_K, and PQ runs use the same thread count,
NUMA mode, generation lengths, warmup policy, and repetition count.

The PQ GGUF currently retains the original F16 tensors for prefill
compatibility. Therefore, its on-disk size is not the same as the decode-time
weight stream. During PQ decode, the llama.cpp loader registers the PQ side
tensors and the runtime selects the PQ GEMV path for one-token decode.

System load, page-cache state, CPU frequency, and NUMA placement can affect
results. For paper numbers, record the complete command, commit IDs, CPU
information, and the CSV output. Do not mix values from different benchmark
configurations in one comparison table.

## 10. Cleanup

```bash
make clean
```

This removes the AE executable. To remove the local llama.cpp build output:

```bash
rm -rf build
```
