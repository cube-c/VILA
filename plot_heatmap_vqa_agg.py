import argparse
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_p_correct(df):
    """P(correct) = p(A) when GT=A, 1-p(A) when GT=B."""
    df = df.copy()
    df["logit_diff"] = df["A_logit"] - df["B_logit"]
    df["p_a"] = 1.0 / (1.0 + np.exp(-df["logit_diff"]))
    gt_a = df["ground_truth"] == "A"
    df["p_correct"] = np.where(gt_a, df["p_a"], 1.0 - df["p_a"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", type=str, nargs="+", required=True,
                        help="Variant CSV files (e.g. *_obj1_closer.csv *_obj2_closer.csv ...)")
    parser.add_argument("--output", "-o", type=str, default="logit_heatmap_vqa_agg.png")
    parser.add_argument("--title", type=str, default="Mean P(correct) across variants (16x16)")
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        df = pd.read_csv(path)
        df = compute_p_correct(df)
        frames.append(df)
        print(f"Loaded {len(df)} rows from {path}")

    df = pd.concat(frames, ignore_index=True)

    # Extract phA and phB
    parsed = df["image"].str.extract(r"phA(\d+)_phB(\d+)\.png$")
    df["phA"] = parsed[0].astype(int)
    df["phB"] = parsed[1].astype(int)

    # Average p_correct across all variants and scenes per cell
    grid_df = df.groupby(["phA", "phB"])["p_correct"].mean().reset_index()
    grid = np.full((16, 16), np.nan)
    for _, row in grid_df.iterrows():
        grid[int(row["phA"]), int(row["phB"])] = row["p_correct"]

    mean_p = np.nanmean(grid)

    # RMSE against average p_correct
    avg_p_correct = df["p_correct"].mean()
    rmse_avg = np.sqrt(np.mean((df["p_correct"] - avg_p_correct) ** 2))

    # Vertical consistency: cols 2-6 vs 10-14
    obj2_lo = np.nanmean(grid[:, 2:7])
    obj2_hi = np.nanmean(grid[:, 10:15])
    vc_obj2 = obj2_hi - obj2_lo

    # Horizontal consistency: rows 2-6 vs 10-14
    obj1_lo = np.nanmean(grid[2:7, :])
    obj1_hi = np.nanmean(grid[10:15, :])
    vc_obj1 = obj1_hi - obj1_lo

    # Wrap-around: rows/cols 6-10 vs (14,15,0,1,2)
    wrap_idx = np.array([14, 15, 0, 1, 2])
    hz_obj1 = np.nanmean(grid[6:11, :]) - np.nanmean(grid[wrap_idx, :])
    hz_obj2 = np.nanmean(grid[:, 6:11]) - np.nanmean(grid[:, wrap_idx])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="equal", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Mean P(correct)")

    metrics_text = (
        f"Mean P(correct) = {mean_p:.4f}\n"
        f"RMSE(avg) = {rmse_avg:.4f}\n"
        f"Obj1 Δ(rows 10-14 vs 2-6)  = {vc_obj1:+.4f}\n"
        f"Obj2 Δ(cols 10-14 vs 2-6)  = {vc_obj2:+.4f}\n"
        f"Obj1 Δ(rows 6-10 vs 14-2)  = {hz_obj1:+.4f}\n"
        f"Obj2 Δ(cols 6-10 vs 14-2)  = {hz_obj2:+.4f}"
    )
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Obj2 Phase (phB)")
    ax.set_ylabel("Obj1 Phase (phA)")
    ax.set_title(args.title)
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
