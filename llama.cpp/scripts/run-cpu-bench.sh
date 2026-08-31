#!/usr/bin/env bash
# Inference performance benchmark for Llama-2-7b-chat on Intel Xeon 6759P-C
# CPU supports AVX512 (full set) and AMX (bf16/int8/tile). Build with:
#   GGML_AMX_BF16/INT8/TILE=ON, GGML_AVX512*=ON, GGML_NATIVE=ON, LTO, Release
#
# Measures prompt processing (pp) and text generation (tg) throughput across
# all GGUF quantization variants. Pinned to NUMA node 0 to avoid cross-socket
# memory traffic, which dominates latency for memory-bound LLM inference.

set -euo pipefail

BENCH="$(cd "$(dirname "$0")/.." && pwd)/build-amx/bin/llama-bench"
MODEL_DIR="/home/shared/models/gguf-models/Llama-2-7b-chat-hf-GGUF"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/bench-results"
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_MD="$OUT_DIR/llama2-7b-$STAMP.md"
OUT_CSV="$OUT_DIR/llama2-7b-$STAMP.csv"

# One NUMA node = 60 cores / 120 threads. LLM inference is memory-bandwidth
# bound; using both sockets rarely helps and hurts via remote memory access.
THREADS="${THREADS:-120}"
PP="${PP:-512}"
TG="${TG:-128}"
REPS="${REPS:-3}"
NUMA="${NUMA:-isolate}"

QUANTS=(
    Q2_K Q3_K_S Q3_K_M Q3_K_L
    Q4_0 Q4_1 Q4_K_S Q4_K_M
    Q5_0 Q5_1 Q5_K_S Q5_K_M
    Q6_K Q8_0
)

echo "Benchmark config: threads=$THREADS  pp=$PP  tg=$TG  reps=$REPS  numa=$NUMA"
echo "Output: $OUT_MD  (csv: $OUT_CSV)"

# llama-bench -o markdown produces a single table; -o csv is row-oriented.
# Run each quant separately so a failure on one does not abort the rest.
for q in "${QUANTS[@]}"; do
    model="$MODEL_DIR/Llama-2-7b-chat-hf.$q.gguf"
    if [[ ! -f "$model" ]]; then
        echo "  [skip] $q: file not found" >&2
        continue
    fi
    echo "  [run ] $q"
    "$BENCH" \
        -m "$model" \
        -p "$PP" -n "$TG" \
        -t "$THREADS" \
        --numa "$NUMA" \
        -r "$REPS" \
        -o md >> "$OUT_MD" 2>&1 || echo "    (failed, see log)" >> "$OUT_MD"
    echo "" >> "$OUT_MD"
done

# CSV variant for downstream analysis (e.g. compare-llama-bench.py)
for q in "${QUANTS[@]}"; do
    model="$MODEL_DIR/Llama-2-7b-chat-hf.$q.gguf"
    [[ -f "$model" ]] || continue
    "$BENCH" \
        -m "$model" \
        -p "$PP" -n "$TG" \
        -t "$THREADS" \
        --numa "$NUMA" \
        -r "$REPS" \
        -o csv >> "$OUT_CSV" 2>&1 || true
done

echo "Done."
echo "  markdown: $OUT_MD"
echo "  csv:      $OUT_CSV"
