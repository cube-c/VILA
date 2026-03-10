import argparse
import json

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


def compute_grid(df):
    """Compute 16x16 grid of mean p_correct."""
    grid_df = df.groupby(["phA", "phB"])["p_correct"].mean().reset_index()
    grid = np.full((16, 16), np.nan)
    for _, row in grid_df.iterrows():
        grid[int(row["phA"]), int(row["phB"])] = row["p_correct"]
    return grid


def plot_heatmap(ax, grid, title):
    im = ax.imshow(grid, cmap="seismic", aspect="equal", vmin=0, vmax=1)
    ax.set_xlabel("Obj2 Phase (phB)")
    ax.set_ylabel("Obj1 Phase (phA)")
    ax.set_title(title)
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", type=str, nargs="+", required=True,
                        help="Variant CSV files (e.g. *_obj1_closer.csv *_obj2_closer.csv ...)")
    parser.add_argument("--cell-class", type=str, required=True,
                        help="Path to cell_class.json")
    parser.add_argument("--output", "-o", type=str, default="logit_heatmap_vqa_counter.png")
    parser.add_argument("--title", type=str, default="Consistent vs Counter")
    args = parser.parse_args()

    # Load cell classification
    with open(args.cell_class) as f:
        raw = json.load(f)
    cell_class = {}
    for key, label in raw.items():
        phA, phB = key.split(",")
        cell_class[(int(phA), int(phB))] = label

    pure_con = {c for c, v in cell_class.items() if v == "consistent"}
    pure_ctr = {c for c, v in cell_class.items() if v == "counter"}
    print(f"Cell classification: consistent={len(pure_con)}, counter={len(pure_ctr)}")

    # Build fraction grid for visualization
    frac_grid = np.full((16, 16), np.nan)
    for cell, label in cell_class.items():
        if label == "consistent":
            frac_grid[cell] = 1.0
        elif label == "counter":
            frac_grid[cell] = 0.0

    # Load and combine CSVs
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

    # Split data by cell classification
    df["_cell"] = list(zip(df["phA"], df["phB"]))
    consistent = df[df["_cell"].isin(pure_con)].copy()
    counter = df[df["_cell"].isin(pure_ctr)].copy()
    print(f"Rows: consistent={len(consistent)}, counter={len(counter)}")

    # Compute grids
    grid_con = compute_grid(consistent)
    grid_ctr = compute_grid(counter)

    # Simple accuracy: mean p_correct per subset
    acc_con = consistent["p_correct"].mean()
    acc_ctr = counter["p_correct"].mean()
    acc_diff = acc_con - acc_ctr

    print(f"\nAccuracy consistent: {acc_con:.4f}")
    print(f"Accuracy counter:    {acc_ctr:.4f}")
    print(f"Difference (con-ctr): {acc_diff:+.4f}")

    # Plot: fraction map | consistent heatmap | counter heatmap | metrics
    fig, axes = plt.subplots(1, 4, figsize=(28, 7),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.6]})
    ax_frac, ax_con, ax_ctr, ax_txt = axes

    # Fraction consistent heatmap (blue=consistent, red=counter)
    im0 = ax_frac.imshow(frac_grid, cmap="RdYlBu", aspect="equal", vmin=0, vmax=1)
    fig.colorbar(im0, ax=ax_frac, label="Fraction consistent", shrink=0.8)
    ax_frac.set_xlabel("Obj2 Phase (phB)")
    ax_frac.set_ylabel("Obj1 Phase (phA)")
    ax_frac.set_title("Cell split: blue=consistent, red=counter")
    ax_frac.set_xticks(range(16))
    ax_frac.set_yticks(range(16))

    im1 = plot_heatmap(ax_con, grid_con, f"Consistent (obj1 higher, n={len(consistent)})")
    fig.colorbar(im1, ax=ax_con, label="Mean P(correct)", shrink=0.8)

    im2 = plot_heatmap(ax_ctr, grid_ctr, f"Counter (obj1 lower, n={len(counter)})")
    fig.colorbar(im2, ax=ax_ctr, label="Mean P(correct)", shrink=0.8)

    # Metrics text
    metrics_text = (
        f"Consistent (n={len(consistent)})\n"
        f"  Mean P(correct) = {acc_con:.4f}\n"
        f"\n"
        f"Counter (n={len(counter)})\n"
        f"  Mean P(correct) = {acc_ctr:.4f}\n"
        f"\n"
        f"Difference (consistent - counter)\n"
        f"  Δ = {acc_diff:+.4f}"
    )
    ax_txt.axis("off")
    ax_txt.text(0.05, 0.95, metrics_text, transform=ax_txt.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved {args.output}")

    # Save TSV
    tsv_path = args.output.rsplit(".", 1)[0] + ".tsv"
    rows = [
        {"metric": "acc_consistent", "value": acc_con},
        {"metric": "acc_counter", "value": acc_ctr},
        {"metric": "acc_diff_con_minus_ctr", "value": acc_diff},
        {"metric": "n_consistent", "value": len(consistent)},
        {"metric": "n_counter", "value": len(counter)},
    ]
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"Saved {tsv_path}")


if __name__ == "__main__":
    main()
