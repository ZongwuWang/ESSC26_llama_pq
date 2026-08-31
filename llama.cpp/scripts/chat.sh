#!/usr/bin/env bash
# Interactive chat with a Llama-2-7b-chat GGUF quant on CPU (AVX512 + AMX).
#
# Usage:
#   ./scripts/chat.sh                # default: Q4_K_M
#   ./scripts/chat.sh Q4_0           # pick a quant
#   ./scripts/chat.sh Q8_0 -n 256    # pass extra llama-cli flags
#   QUANT=Q6_K ./scripts/chat.sh     # or via env
#
# Quant options:
#   Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_0 Q4_1 Q4_K_S Q4_K_M
#   Q5_0 Q5_1 Q5_K_S Q5_K_M Q6_K Q8_0

set -euo pipefail

CLI="$(cd "$(dirname "$0")/.." && pwd)/build-amx/bin/llama-cli"
MODEL_DIR="/home/shared/models/gguf-models/Llama-2-7b-chat-hf-GGUF"

QUANT="${1:-${QUANT:-Q4_K_M}}"; shift || true
EXTRA_ARGS=("$@")

# 120 threads = one NUMA node (60 cores x 2 SMT). isolate keeps memory local.
THREADS="${THREADS:-120}"
NUMA="${NUMA:-isolate}"
CTX="${CTX:-4096}"
N_GEN="${N_GEN:--1}"   # -1 = until context full / user stops

model="$MODEL_DIR/Llama-2-7b-chat-hf.$QUANT.gguf"
if [[ ! -f "$model" ]]; then
    echo "Model not found: $model" >&2
    echo "Available quants:" >&2
    ls "$MODEL_DIR"/*.gguf | sed 's/.*\.\(Q[^.]*\)\.gguf/  \1/' >&2
    exit 1
fi

echo ">>> quant=$QUANT  threads=$THREADS  numa=$NUMA  ctx=$CTX"
echo ">>> model: $model"
echo ">>> type /exit or Ctrl+C to quit, /clear to reset history"
echo

exec "$CLI" \
    -m "$model" \
    -t "$THREADS" \
    --numa "$NUMA" \
    -c "$CTX" \
    -n "$N_GEN" \
    -cnv \
    --display-prompt \
    "${EXTRA_ARGS[@]}"
