#!/usr/bin/env python3
"""Prepare WikiText-2 and evaluate GGUF or PQ-checkpoint perplexity."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path


CSV_FIELDS = [
    "model", "path", "dataset", "split", "context", "stride",
    "threads", "numa", "ppl", "method",
]


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("model must use NAME=PATH")
    return name, path


def load_test_text(dataset_path: Path) -> str:
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))["test"]
    return "\n\n".join(dataset["text"])


def prepare_dataset(output: Path) -> None:
    from datasets import load_dataset

    if output.exists():
        print(f"[REUSE] WikiText-2 dataset: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    )
    dataset.save_to_disk(str(output))
    print(f"saved WikiText-2 to {output}")


def run_and_capture(command: list[str]) -> str:
    print("$ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.pop("GGML_NODE_TIMING", None)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}")
    return "".join(lines)


def parse_ppl(output: str) -> float:
    patterns = [
        r"perplexity:\s*([0-9]+(?:\.[0-9]+)?)",
        r"\[\d+\]([0-9]+(?:\.[0-9]+)?)",
        r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            return float(matches[-1])
    raise RuntimeError("could not parse perplexity from command output")


def write_rows(output: Path, rows: list[dict[str, str]], append: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if append and output.exists():
        with output.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
        replaced = {row["model"] for row in rows}
        existing = [row for row in existing if row["model"] not in replaced]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing + rows)
    print(f"wrote {output}")


def evaluate_gguf(args: argparse.Namespace) -> None:
    binary = Path(args.binary)
    dataset = Path(args.dataset)
    if not binary.is_file():
        raise SystemExit(f"perplexity binary does not exist: {binary}")
    if not dataset.is_dir():
        raise SystemExit(f"dataset does not exist: {dataset}")

    temporary = None
    if args.text:
        text_path = Path(args.text)
        text_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="edgepq-ppl-")
        text_path = Path(temporary.name) / "wikitext-2-raw-v1.test.txt"
    text_path.write_text(load_test_text(dataset), encoding="utf-8")

    rows: list[dict[str, str]] = []
    for name, model_name in args.model:
        model = Path(model_name)
        if not model.is_file():
            raise SystemExit(f"model does not exist: {model}")
        command = [
            str(binary), "-m", str(model), "-f", str(text_path),
            "-c", str(args.context), "--ppl-stride", str(args.stride),
            "-t", str(args.threads), "--numa", args.numa,
            "--device", args.device, "--no-repack", "--no-perf",
        ]
        ppl = parse_ppl(run_and_capture(command))
        print(f"[{name}] PPL = {ppl:.4f}")
        rows.append({
            "model": name,
            "path": str(model),
            "dataset": str(dataset),
            "split": "test",
            "context": str(args.context),
            "stride": str(args.stride),
            "threads": str(args.threads),
            "numa": args.numa,
            "ppl": f"{ppl:.4f}",
            "method": "GGUF GPU llama-perplexity",
        })
    write_rows(Path(args.output), rows, append=False)
    if temporary is not None:
        temporary.cleanup()


def llama2_7b_config():
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
    )


def unwrap_state_dict(value):
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict", "non_pq_state"):
            candidate = value.get(key)
            if isinstance(candidate, dict):
                return candidate
    if not isinstance(value, dict):
        raise TypeError("non_pq_state.pt must contain a state dictionary")
    return value


def pq_parameter_name(path: Path) -> str:
    return path.stem.replace("__", ".") + ".weight"


def reconstruct_pq_weight(state: dict) -> "torch.Tensor":
    import torch

    if state.get("format") != "pq-4c8b":
        raise ValueError("unsupported PQ state format")
    codebooks = state["codebooks"].float().cpu()
    codes = state["codes"].long().cpu()
    block_map = state["block_to_codebook"].long().cpu()
    scales = state["scales"].float().reshape(-1).cpu()
    vector_scales = state["vector_scales"].float().reshape(-1).cpu()
    dimension_scales = state["dimension_scales"].float().reshape(-1).cpu()

    blocks, n_out, subspaces = codes.shape
    width = subspaces * 4
    result = torch.empty((n_out, blocks * width), dtype=torch.float16)
    subspace_ids = torch.arange(subspaces).view(1, -1)

    for block in range(blocks):
        table = codebooks[int(block_map[block])]
        selected = table[subspace_ids, codes[block]].reshape(n_out, width)
        begin = block * width
        end = begin + width
        selected.mul_(scales[block])
        selected.mul_(dimension_scales[begin:end].view(1, -1))
        selected.mul_(vector_scales.view(-1, 1))
        result[:, begin:end] = selected.to(torch.float16)
    return result


def evaluate_pq(args: argparse.Namespace) -> None:
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoTokenizer, LlamaForCausalLM

    checkpoint = Path(args.checkpoint)
    dataset = Path(args.dataset)
    non_pq_path = checkpoint / "non_pq_state.pt"
    pq_dir = checkpoint / "pq_states"
    if not non_pq_path.is_file() or not pq_dir.is_dir():
        raise SystemExit(f"incomplete PQ checkpoint: {checkpoint}")
    if not dataset.is_dir():
        raise SystemExit(f"dataset does not exist: {dataset}")
    if args.stride != args.context:
        raise SystemExit("the paper PQ protocol requires stride == context")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    with init_empty_weights():
        model = LlamaForCausalLM(llama2_7b_config())

    non_pq = unwrap_state_dict(torch.load(non_pq_path, map_location="cpu", weights_only=False))
    for name, tensor in non_pq.items():
        normalized = name.removeprefix("module.")
        set_module_tensor_to_device(model, normalized, "cpu", value=tensor.to(torch.float16))

    pq_files = sorted(pq_dir.glob("*.pt"))
    if len(pq_files) != 224:
        raise RuntimeError(f"expected 224 PQ states, found {len(pq_files)}")
    for index, path in enumerate(pq_files, start=1):
        state = torch.load(path, map_location="cpu", weights_only=False)
        set_module_tensor_to_device(
            model,
            pq_parameter_name(path),
            "cpu",
            value=reconstruct_pq_weight(state),
        )
        if index == 1 or index % 16 == 0 or index == len(pq_files):
            percentage = 100.0 * index / len(pq_files)
            print(
                f"[EdgePQ setup] reconstructed {index}/{len(pq_files)} "
                f"layers ({percentage:.0f}%)",
                flush=True,
            )

    missing = [name for name, parameter in model.named_parameters() if parameter.device.type == "meta"]
    if missing:
        raise RuntimeError(f"checkpoint left {len(missing)} parameters uninitialized: {missing[:8]}")

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    input_ids = tokenizer(load_test_text(dataset), return_tensors="pt")["input_ids"]
    total_nll = 0.0
    total_tokens = 0
    windows = 0
    previous_end = 0
    total_windows = input_ids.size(1) // args.context

    with torch.inference_mode():
        for begin in range(0, input_ids.size(1), args.stride):
            end = min(begin + args.context, input_ids.size(1))
            target_length = end - previous_end
            if target_length != args.context:
                continue
            tokens = input_ids[:, begin:end].to(device)
            labels = tokens.clone()
            labels[:, :-target_length] = -100
            loss = model(tokens, labels=labels).loss
            total_nll += float(loss) * target_length
            total_tokens += target_length
            windows += 1
            previous_end = end
            if windows == 1 or windows % 5 == 0 or windows == total_windows:
                percentage = 100.0 * windows / total_windows
                print(
                    f"[EdgePQ PPL] {windows}/{total_windows} windows "
                    f"({percentage:.0f}%) tokens={total_tokens} "
                    f"ppl={math.exp(total_nll / total_tokens):.4f}",
                    flush=True,
                )
            if end == input_ids.size(1):
                break

    if not total_tokens:
        raise RuntimeError("no complete PQ PPL window was produced")

    ppl = math.exp(total_nll / total_tokens)
    rows = [{
        "model": "EdgePQ",
        "path": str(checkpoint),
        "dataset": str(dataset),
        "split": "test",
        "context": str(args.context),
        "stride": str(args.stride),
        "threads": "NA",
        "numa": "NA",
        "ppl": f"{ppl:.4f}",
        "method": "complete-window PQ-to-FP16 reconstruction",
    }]
    print(f"[EdgePQ] windows={windows} tokens={total_tokens} PPL = {ppl:.4f}")
    write_rows(Path(args.output), rows, append=args.append)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-dataset", help="download and save WikiText-2")
    prepare.add_argument("--output", required=True)

    gguf = subparsers.add_parser("gguf", help="evaluate F16/Q2_K GGUF perplexity")
    gguf.add_argument("--binary", required=True)
    gguf.add_argument("--dataset", required=True)
    gguf.add_argument("--text", help="optional persistent WikiText-2 text file")
    gguf.add_argument("--output", default="output/perplexity.csv")
    gguf.add_argument("--context", type=int, default=4096)
    gguf.add_argument("--stride", type=int, default=4096)
    gguf.add_argument("--threads", type=int, default=60)
    gguf.add_argument("--numa", default="distribute")
    gguf.add_argument("--device", default="CUDA0")
    gguf.add_argument("--model", action="append", required=True, type=parse_model)

    pq = subparsers.add_parser("pq", help="evaluate PQ checkpoint reconstruction perplexity")
    pq.add_argument("--checkpoint", required=True)
    pq.add_argument("--dataset", required=True)
    pq.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    pq.add_argument("--output", default="output/perplexity.csv")
    pq.add_argument("--context", type=int, default=4096)
    pq.add_argument("--stride", type=int, default=4096)
    pq.add_argument("--device", default="cuda:0")
    pq.add_argument("--append", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-dataset":
        prepare_dataset(Path(args.output))
    elif args.command == "gguf":
        evaluate_gguf(args)
    elif args.command == "pq":
        evaluate_pq(args)


if __name__ == "__main__":
    main()
