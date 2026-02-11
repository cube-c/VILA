import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def aggregate(input_path, output_path):
    # Accumulate sums and counts per (model, task, layer_idx)
    stats = defaultdict(lambda: {"image_ratio": 0.0, "prompt_ratio": 0.0, "gen_ratio": 0.0, "count": 0})

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["task"], int(row["layer_idx"]))
            stats[key]["image_ratio"] += float(row["image_ratio"])
            stats[key]["prompt_ratio"] += float(row["prompt_ratio"])
            stats[key]["gen_ratio"] += float(row["gen_ratio"])
            stats[key]["count"] += 1

    # Build aggregated rows sorted by (model, task, layer_idx)
    fieldnames = ["model", "task", "layer_idx", "count", "avg_image_ratio", "avg_prompt_ratio", "avg_gen_ratio"]
    rows = []
    for (model, task, layer_idx), s in sorted(stats.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        n = s["count"]
        rows.append({
            "model": model,
            "task": task,
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

    print(f"Aggregated {sum(s['count'] for s in stats.values())} rows into {len(rows)} (model, task, layer) groups")
    print(f"CSV output: {output_path}")
    return rows


def plot(rows, output_path):
    # Group rows by (model, task)
    groups = {}
    for row in rows:
        key = (row["model"], row["task"])
        if key not in groups:
            groups[key] = {"layers": [], "image": [], "prompt": [], "gen": []}
        groups[key]["layers"].append(row["layer_idx"])
        groups[key]["image"].append(row["avg_image_ratio"])
        groups[key]["prompt"].append(row["avg_prompt_ratio"])
        groups[key]["gen"].append(row["avg_gen_ratio"])

    group_keys = sorted(groups.keys())
    n_groups = len(group_keys)

    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 5 * n_groups), squeeze=False)

    for i, (model_name, task) in enumerate(group_keys):
        ax = axes[i, 0]
        data = groups[(model_name, task)]
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
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Attention Ratio")
        ax.set_title(f"{model_name} — {task}")
        ax.set_xlim(layers.min(), layers.max())
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Avg Attention Ratios by Layer (CV-Bench 2D)", fontsize=14, y=1.01)
    fig.tight_layout()

    plot_path = output_path.replace(".csv", ".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close(fig)


def plot_image_ratio_comparison(rows, output_path):
    # Group rows by (task, model)
    by_task = defaultdict(dict)
    for row in rows:
        task, model = row["task"], row["model"]
        if model not in by_task[task]:
            by_task[task][model] = {"layers": [], "image": []}
        by_task[task][model]["layers"].append(row["layer_idx"])
        by_task[task][model]["image"].append(row["avg_image_ratio"])

    def short_name(model):
        return model.split("/")[-1]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    styles = ["-", "--", "-.", ":"]

    tasks = sorted(by_task.keys())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(12, 5 * len(tasks)), squeeze=False)

    for t, task in enumerate(tasks):
        ax = axes[t, 0]
        models = sorted(by_task[task].keys())
        for i, model in enumerate(models):
            data = by_task[task][model]
            layers = np.array(data["layers"])
            image = np.nan_to_num(np.array(data["image"]))
            ax.plot(layers, image, label=short_name(model),
                    color=colors[i % len(colors)],
                    linestyle=styles[i % len(styles)],
                    linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Avg Image Attention Ratio")
        ax.set_title(f"Image Attention Ratio — {task}")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Image Attention Ratio Comparison (CV-Bench 2D)", fontsize=14, y=1.01)
    fig.tight_layout()

    plot_path = output_path.replace(".csv", "_image_compare.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate attention ratios per model, task, and layer index")
    parser.add_argument("--input", type=str, default="output/attention_ratios_cvbench.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default="output/attention_ratios_cvbench_agg.csv", help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows = aggregate(args.input, args.output)
    plot(rows, args.output)
    plot_image_ratio_comparison(rows, args.output)


if __name__ == "__main__":
    main()
