import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Canonical model order and display labels
MODEL_ORDER = ["base", "80k", "400k", "800k", "2m", "roborefer"]
MODEL_LABELS = {
    "base":      "Base",
    "80k":       "80K",
    "400k":      "400K",
    "800k":      "800K",
    "2m":        "2M",
    "roborefer": "RoboRefer",
}

SPLIT_ORDER = ["h11", "h12", "h13"]
SPLIT_COLORS = {"h11": "#4C72B0", "h12": "#DD8452", "h13": "#55A868"}
SPLIT_LABELS = {"h11": "small", "h12": "medium", "h13": "large"}


def parse_filename(path, prefix=""):
    """Return (model_type, split) from a logit_results_*.csv filename, or None.

    prefix: optional infix after 'logit_results_' that identifies the family
            (e.g. 'molmo').  Files not matching the prefix are ignored.
    """
    name = os.path.splitext(os.path.basename(path))[0]  # strip .csv
    p = f"logit_results_{prefix}_" if prefix else "logit_results_"

    if not name.startswith(p[:-1]):   # quick reject
        return None

    # Strip the prefix to get the remainder
    rest = name[len(p):]   # e.g. "h11", "80k_h11", "roborefer_h11"

    # roborefer (only without a model-family prefix)
    if not prefix:
        m = re.fullmatch(r"roborefer(?:_(h\d+))?", rest)
        if m:
            return ("roborefer", m.group(1))

    # base: just hNN
    m = re.fullmatch(r"(h\d+)", rest)
    if m:
        return ("base", m.group(1))

    # scale_hNN  (80k / 400k / 800k / 2m)
    m = re.fullmatch(r"(\w+?)_(h\d+)", rest)
    if m:
        return (m.group(1), m.group(2))

    return None


def load_all(csv_dir, prefix=""):
    rows = []
    for path in sorted(glob.glob(os.path.join(csv_dir, "logit_results_*.csv"))):
        parsed = parse_filename(path, prefix)
        if parsed is None:
            continue
        model_type, split = parsed
        if split is None:          # files without a split suffix — skip
            continue
        if model_type not in MODEL_ORDER:
            continue
        df = pd.read_csv(path)
        df["logit_diff"] = df["Yes_logit"] - df["No_logit"]
        df["model_type"] = model_type
        df["split"] = split
        rows.append(df)

    if not rows:
        raise RuntimeError("No matching CSV files found.")
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", "-d", type=str, default=".",
                        help="Directory containing logit_results_*.csv files")
    parser.add_argument("--output", "-o", type=str, default="logit_stats.png")
    parser.add_argument("--title", type=str, default="Logit Diff (Yes − No) by Model & Split")
    parser.add_argument("--prefix", type=str, default="",
                        help="Filename family prefix, e.g. 'molmo' for logit_results_molmo_*.csv")
    args = parser.parse_args()

    df = load_all(args.csv_dir, args.prefix)

    stats = (
        df.groupby(["model_type", "split"])["logit_diff"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    n_models = len(MODEL_ORDER)
    n_splits = len(SPLIT_ORDER)
    bar_w = 0.22
    group_gap = 0.9
    x_centers = np.arange(n_models) * group_gap

    fig, ax = plt.subplots(figsize=(10, 5))

    for si, split in enumerate(SPLIT_ORDER):
        offsets = (si - (n_splits - 1) / 2) * bar_w
        heights, errs, xs = [], [], []
        for mi, model in enumerate(MODEL_ORDER):
            row = stats[(stats["model_type"] == model) & (stats["split"] == split)]
            if row.empty:
                heights.append(0.0)
                errs.append(0.0)
            else:
                heights.append(row["mean"].iloc[0])
                errs.append(row["std"].iloc[0])
            xs.append(x_centers[mi] + offsets)

        ax.bar(xs, heights, width=bar_w, label=SPLIT_LABELS[split],
               color=SPLIT_COLORS[split], alpha=0.85, edgecolor="white")
        ax.errorbar(xs, heights, yerr=errs, fmt="none",
                    ecolor="black", elinewidth=1, capsize=3)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("Logit diff (Yes − No)")
    ax.set_title(args.title)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.legend(title="Split")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
