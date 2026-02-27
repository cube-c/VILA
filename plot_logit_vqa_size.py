import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_sizes(image_path):
    """Extract (s1, s2) from filename like '0000_s10.100_s20.300.png'."""
    m = re.search(r"s1([\d.]+)_s2([\d.]+)\.png$", image_path)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="logit_results_vqa.csv (single CSV)")
    parser.add_argument("--output", "-o", type=str, default="logit_vqa_size.png")
    parser.add_argument("--title", type=str, default="Logit Diff (Yes − No) by Size Pair")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["logit_diff"] = df["Yes_logit"] - df["No_logit"]
    sizes = df["image"].apply(lambda p: pd.Series(extract_sizes(p), index=["s1", "s2"]))
    df = pd.concat([df, sizes], axis=1)
    df = df.dropna(subset=["s1", "s2"])
    df["size_diff"] = df["s1"] - df["s2"]

    stats = (
        df.groupby(["s1", "s2"])["logit_diff"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    stats = stats.sort_values("s1").reset_index(drop=True)

    labels = [f"{row.s1:.2f} / {row.s2:.2f}" for _, row in stats.iterrows()]
    x = np.arange(len(stats))

    # Color bars by whether obj1 is smaller, equal, or larger than obj2
    colors = []
    for _, row in stats.iterrows():
        if row.s1 < row.s2:
            colors.append("#348ABD")   # obj1 smaller
        elif row.s1 > row.s2:
            colors.append("#E24A33")   # obj1 larger
        else:
            colors.append("#888888")   # equal

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x, stats["mean"], width=0.7, color=colors, alpha=0.85, edgecolor="white")
    ax.errorbar(x, stats["mean"], yerr=stats["std"], fmt="none",
                ecolor="black", elinewidth=1, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Obj1 size / Obj2 size")
    ax.set_ylabel("Logit diff (Yes − No)")
    ax.set_title(args.title)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

    # Legend for colors
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#348ABD", label="obj1 < obj2"),
        Patch(color="#888888", label="obj1 = obj2"),
        Patch(color="#E24A33", label="obj1 > obj2"),
    ], title="Size relation")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
