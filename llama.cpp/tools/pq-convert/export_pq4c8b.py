#!/usr/bin/env python3
"""Export TFLOP pq-4c8b states into llama-pq-convert side buffers."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch


def gguf_name(path: Path) -> str:
    parts = path.stem.split("__")
    if len(parts) < 5 or parts[0] != "model":
        raise ValueError(path)
    layer = parts[2]
    group = parts[3]
    proj = parts[4]
    group_name = {"self_attn": "attn", "mlp": "ffn"}[group]
    proj_name = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "output",
                 "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}[proj]
    return f"blk.{layer}.{group_name}_{proj_name}.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.checkpoint) / "pq_states"
    count = 0
    for src in sorted(root.glob("*.pt")):
        state = torch.load(src, map_location="cpu", weights_only=False)
        if state.get("format") != "pq-4c8b":
            continue
        if state["axis"] != 1 or state["couples"] != 4 or state["n_bits"] != 8:
            raise ValueError(f"unsupported state {src}")
        cb = state["codebooks"].cpu().numpy().astype(np.float32)
        codes = state["codes"].cpu().numpy()
        block_map = state["block_to_codebook"].cpu().numpy()
        scales = state["scales"].cpu().numpy().astype(np.float32)
        vscale = state["vector_scales"].cpu().numpy().reshape(-1).astype(np.float32)
        dscale = state["dimension_scales"].cpu().numpy().reshape(-1).astype(np.float32)
        blocks, n_out, sub = codes.shape
        if cb.shape[:1] != (blocks,) or cb.shape[1:] != (sub, 256, 4):
            raise ValueError(f"shape mismatch {src}")
        n_in = sub * 4 * blocks
        cb_out = np.empty((n_in, 256), dtype=np.float16)
        for b in range(blocks):
            table = cb[int(block_map[b])]
            for s in range(sub):
                col = (b * sub + s) * 4
                cb_out[col:col+4] = (table[s].T * (scales[b] * dscale[col:col+4, None])).astype(np.float16)
        idx_out = np.transpose(codes, (0, 2, 1)).reshape(-1).astype(np.uint8)
        name = gguf_name(src)
        cb_out.tofile(out / (name + ".cb"))
        idx_out.tofile(out / (name + ".idx"))
        vscale.tofile(out / (name + ".scale"))
        count += 1
    print(f"exported {count} pq-4c8b tensors to {out}")

if __name__ == "__main__":
    main()
