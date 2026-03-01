import argparse
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_p_correct(df):
    """P(correct) = p(A) when GT=A, 1-p(A) when GT=B."""
    df = df.copy()
    df["logit_diff"] = df["Yes_logit"] - df["No_logit"]
    df["p_yes"] = 1.0 / (1.0 + np.exp(-df["logit_diff"]))
    gt_yes = df["ground_truth"] == "Yes"
    df["p_correct"] = np.where(gt_yes, df["p_yes"], 1.0 - df["p_yes"])
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
    hz_obj1_mid = np.nanmean(grid[6:11, :])
    hz_obj1_wrap = np.nanmean(grid[wrap_idx, :])
    hz_obj1 = hz_obj1_mid - hz_obj1_wrap
    hz_obj2_mid = np.nanmean(grid[:, 6:11])
    hz_obj2_wrap = np.nanmean(grid[:, wrap_idx])
    hz_obj2 = hz_obj2_mid - hz_obj2_wrap

    # Overlapping corners: both objects in specific regions
    o1top_o2bot = np.nanmean(grid[2:7, 10:15])    # obj1 top, obj2 bottom
    o1bot_o2top = np.nanmean(grid[10:15, 2:7])    # obj1 bottom, obj2 top
    overlap_delta = o1top_o2bot - o1bot_o2top

    from matplotlib.patches import Rectangle

    fig, (ax, ax_txt) = plt.subplots(1, 2, figsize=(14, 7),
                                      gridspec_kw={"width_ratios": [1, 0.5]})
    im = ax.imshow(grid, cmap="seismic", aspect="equal", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Mean P(correct)")

    # Cyan bounding boxes for overlap regions
    # O1top+O2bot: rows 2-6, cols 10-14 → rect at (col=9.5, row=1.5) size 5x5
    ax.add_patch(Rectangle((9.5, 1.5), 5, 5, linewidth=2, edgecolor="cyan", facecolor="none"))
    # O1bot+O2top: rows 10-14, cols 2-6 → rect at (col=1.5, row=9.5) size 5x5
    ax.add_patch(Rectangle((1.5, 9.5), 5, 5, linewidth=2, edgecolor="cyan", facecolor="none"))

    ax.set_xlabel("Obj2 Phase (phB)")
    ax.set_ylabel("Obj1 Phase (phA)")
    ax.set_title(args.title)
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))

    metrics_text = (
        f"Mean P(correct) = {mean_p:.4f}\n"
        f"RMSE(avg) = {rmse_avg:.4f}\n"
        f"\n"
        f"Obj1 top={obj1_lo:.4f}  bottom={obj1_hi:.4f}  Δ={vc_obj1:+.4f}\n"
        f"Obj2 top={obj2_lo:.4f}  bottom={obj2_hi:.4f}  Δ={vc_obj2:+.4f}\n"
        f"\n"
        f"Obj1 right={hz_obj1_wrap:.4f}  left={hz_obj1_mid:.4f}  Δ={hz_obj1:+.4f}\n"
        f"Obj2 right={hz_obj2_wrap:.4f}  left={hz_obj2_mid:.4f}  Δ={hz_obj2:+.4f}\n"
        f"\n"
        f"O1top+O2bot = {o1top_o2bot:.4f}\n"
        f"O1bot+O2top = {o1bot_o2top:.4f}\n"
        f"Δ           = {overlap_delta:+.4f}"
    )
    ax_txt.axis("off")
    ax_txt.text(0.05, 0.95, metrics_text, transform=ax_txt.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")

    # Save metrics as TSV
    tsv_path = args.output.rsplit(".", 1)[0] + ".tsv"
    metrics_rows = [
        {"metric": "mean_p_correct", "value": mean_p},
        {"metric": "rmse_avg", "value": rmse_avg},
        {"metric": "obj1_top", "value": obj1_lo},
        {"metric": "obj1_bottom", "value": obj1_hi},
        {"metric": "obj1_delta_bottom_vs_top", "value": vc_obj1},
        {"metric": "obj2_top", "value": obj2_lo},
        {"metric": "obj2_bottom", "value": obj2_hi},
        {"metric": "obj2_delta_bottom_vs_top", "value": vc_obj2},
        {"metric": "obj1_right", "value": hz_obj1_wrap},
        {"metric": "obj1_left", "value": hz_obj1_mid},
        {"metric": "obj1_delta_left_vs_right", "value": hz_obj1},
        {"metric": "obj2_right", "value": hz_obj2_wrap},
        {"metric": "obj2_left", "value": hz_obj2_mid},
        {"metric": "obj2_delta_left_vs_right", "value": hz_obj2},
        {"metric": "o1top_o2bot", "value": o1top_o2bot},
        {"metric": "o1bot_o2top", "value": o1bot_o2top},
        {"metric": "delta_o1top_o2bot_vs_o1bot_o2top", "value": overlap_delta},
    ]
    pd.DataFrame(metrics_rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"Saved {tsv_path}")


if __name__ == "__main__":
    main()
