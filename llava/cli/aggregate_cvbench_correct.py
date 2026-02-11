"""Aggregate attention ratios from the parsed CV-Bench CSV, grouped by correctness.

Groups by (model, task, correct, layer_idx) and produces:
  - CSV with avg attention ratios
  - Stacked area plots per (model, task, correct)
  - Image ratio comparison: correct vs incorrect per (model, task)
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def aggregate(input_path, output_path):
    stats = defaultdict(lambda: {"image_ratio": 0.0, "prompt_ratio": 0.0, "gen_ratio": 0.0, "count": 0})

    skipped = 0
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            correct = row.get("correct", "")
            if correct == "":
                skipped += 1
                continue
            key = (row["model"], row["task"], correct, int(row["layer_idx"]))
            stats[key]["image_ratio"] += float(row["image_ratio"])
            stats[key]["prompt_ratio"] += float(row["prompt_ratio"])
            stats[key]["gen_ratio"] += float(row["gen_ratio"])
            stats[key]["count"] += 1

    fieldnames = ["model", "task", "correct", "layer_idx", "count", "avg_image_ratio", "avg_prompt_ratio", "avg_gen_ratio"]
    rows = []
    for (model, task, correct, layer_idx), s in sorted(stats.items()):
        n = s["count"]
        rows.append({
            "model": model,
            "task": task,
            "correct": correct,
            "layer_idx": layer_idx,
            "count": n,
            "avg_image_ratio": s["image_ratio"] / n,
            "avg_prompt_ratio": s["prompt_ratio"] / n,
            "avg_gen_ratio": s["gen_ratio"] / n,
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(s["count"] for s in stats.values())
    print(f"Aggregated {total} rows into {len(rows)} (model, task, correct, layer) groups")
    print(f"Skipped {skipped} rows with no predicted answer")
    print(f"CSV output: {output_path}")
    return rows


def plot_stacked(rows, output_path):
    """Stacked area plot per (model, task, correct)."""
    groups = {}
    for row in rows:
        key = (row["model"], row["task"], row["correct"])
        if key not in groups:
            groups[key] = {"layers": [], "image": [], "prompt": [], "gen": []}
        groups[key]["layers"].append(row["layer_idx"])
        groups[key]["image"].append(row["avg_image_ratio"])
        groups[key]["prompt"].append(row["avg_prompt_ratio"])
        groups[key]["gen"].append(row["avg_gen_ratio"])

    group_keys = sorted(groups.keys())
    n_groups = len(group_keys)

    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 4 * n_groups), squeeze=False)

    for i, (model, task, correct) in enumerate(group_keys):
        ax = axes[i, 0]
        data = groups[(model, task, correct)]
        layers = np.array(data["layers"])
        image = np.nan_to_num(np.array(data["image"]))
        prompt = np.nan_to_num(np.array(data["prompt"]))
        gen = np.nan_to_num(np.array(data["gen"]))

        ax.stackplot(
            layers, image, prompt, gen,
            labels=["Image", "Prompt", "Generated"],
            colors=["#4C72B0", "#55A868", "#C44E52"],
            alpha=0.85,
        )
        label = "Correct" if correct == "True" else "Incorrect"
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Attention Ratio")
        ax.set_title(f"{model.split('/')[-1]} — {task} — {label}")
        ax.set_xlim(layers.min(), layers.max())
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Avg Attention Ratios by Layer & Correctness (CV-Bench 2D)", fontsize=14, y=1.01)
    fig.tight_layout()

    plot_path = output_path.replace(".csv", ".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close(fig)


def plot_correct_vs_incorrect(rows, output_path):
    """Compare image attention ratio for correct vs incorrect per (model, task)."""
    by_group = defaultdict(lambda: defaultdict(lambda: {"layers": [], "image": []}))
    for row in rows:
        key = (row["model"], row["task"])
        label = "Correct" if row["correct"] == "True" else "Incorrect"
        by_group[key][label]["layers"].append(row["layer_idx"])
        by_group[key][label]["image"].append(row["avg_image_ratio"])

    group_keys = sorted(by_group.keys())
    n_groups = len(group_keys)

    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 5 * n_groups), squeeze=False)

    colors = {"Correct": "#55A868", "Incorrect": "#C44E52"}

    for i, (model, task) in enumerate(group_keys):
        ax = axes[i, 0]
        for label in ["Correct", "Incorrect"]:
            if label not in by_group[(model, task)]:
                continue
            data = by_group[(model, task)][label]
            layers = np.array(data["layers"])
            image = np.nan_to_num(np.array(data["image"]))
            ax.plot(layers, image, label=label,
                    color=colors[label], linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Avg Image Attention Ratio")
        ax.set_title(f"{model.split('/')[-1]} — {task}")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Image Attention: Correct vs Incorrect (CV-Bench 2D)", fontsize=14, y=1.01)
    fig.tight_layout()

    plot_path = output_path.replace(".csv", "_correct_compare.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate attention ratios by correctness")
    parser.add_argument("--input", type=str, default="output/attention_ratios_cvbench_parsed.csv")
    parser.add_argument("--output", type=str, default="output/attention_ratios_cvbench_correct_agg.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows = aggregate(args.input, args.output)
    plot_stacked(rows, args.output)
    plot_correct_vs_incorrect(rows, args.output)


if __name__ == "__main__":
    main()
