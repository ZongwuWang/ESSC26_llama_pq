#!/usr/bin/env python3
"""Render the EdgePQ throughput and perplexity CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing CSV: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def display_model(name: str) -> str:
    return "EdgePQ" if name == "PQ-4c8b" else name


def print_summary(
    throughput_rows: list[dict[str, str]],
    perplexity_rows: list[dict[str, str]],
) -> None:
    generations = sorted({int(row["generation"]) for row in throughput_rows})
    models = list(dict.fromkeys(row["model"] for row in throughput_rows))

    print("\n=== Paper Results Summary ===")
    print("\nDecode throughput (tokens/s, mean +/- stdev)")
    column_width = 18
    header = f"{'Model':<10}" + "".join(
        f"{f'TG{generation}':>{column_width}}" for generation in generations
    )
    print(header)
    print("-" * len(header))
    for model in models:
        by_generation = {
            int(row["generation"]): row
            for row in throughput_rows
            if row["model"] == model
        }
        formatted_values = []
        for generation in generations:
            row = by_generation[generation]
            value = (
                f"{float(row['tokens_per_second']):.2f} +/- "
                f"{float(row['stdev']):.2f}"
            )
            formatted_values.append(f"{value:>{column_width}}")
        values = "".join(formatted_values)
        print(f"{display_model(model):<10}{values}")

    print("\nWikiText-2 perplexity (lower is better)")
    print(f"{'Model':<10}{'PPL':>8}")
    print("-" * 18)
    for row in perplexity_rows:
        print(f"{display_model(row['model']):<10}{float(row['ppl']):>8.4f}")
    print()


def plot_throughput(rows: list[dict[str, str]], output: Path) -> None:
    models = list(dict.fromkeys(row["model"] for row in rows))
    generations = sorted({int(row["generation"]) for row in rows})
    x = np.arange(len(generations))
    width = 0.8 / len(models)

    fig, axis = plt.subplots(figsize=(6.8, 4.0))
    for index, model in enumerate(models):
        model_rows = {int(row["generation"]): row for row in rows if row["model"] == model}
        values = [float(model_rows[g]["tokens_per_second"]) for g in generations]
        errors = [float(model_rows[g]["stdev"]) for g in generations]
        offset = (index - (len(models) - 1) / 2) * width
        axis.bar(x + offset, values, width, yerr=errors, capsize=3, label=display_model(model))

    axis.set_xticks(x, [f"TG{generation}" for generation in generations])
    axis.set_ylabel("Tokens/s")
    axis.set_title("Llama-2-7B decode throughput")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_perplexity(rows: list[dict[str, str]], output: Path) -> None:
    models = [display_model(row["model"]) for row in rows]
    values = [float(row["ppl"]) for row in rows]

    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    bars = axis.bar(models, values)
    axis.set_ylabel("Perplexity (lower is better)")
    axis.set_title("Llama-2-7B WikiText-2 perplexity")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                  ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throughput", default="output/throughput.csv")
    parser.add_argument("--perplexity", default="output/perplexity.csv")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    throughput_rows = read_csv(Path(args.throughput))
    perplexity_rows = read_csv(Path(args.perplexity))
    plot_throughput(throughput_rows, output_dir / "throughput.png")
    plot_perplexity(perplexity_rows, output_dir / "ppl.png")
    print(f"wrote {output_dir / 'throughput.png'}")
    print(f"wrote {output_dir / 'ppl.png'}")
    print_summary(throughput_rows, perplexity_rows)


if __name__ == "__main__":
    main()
