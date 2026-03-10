import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = ["obj1_closer", "obj2_closer", "obj1_farther", "obj2_farther"]


def compute_p_correct(df):
    df = df.copy()
    df["logit_diff"] = df["Yes_logit"] - df["No_logit"]
    df["p_yes"] = 1.0 / (1.0 + np.exp(-df["logit_diff"]))
    gt_yes = df["ground_truth"] == "Yes"
    df["p_correct"] = np.where(gt_yes, df["p_yes"], 1.0 - df["p_yes"])
    return df


def load_grid(csv_prefix):
    """Load all 4 variant CSVs for a model and return 16x16 mean accuracy grid."""
    frames = []
    for v in VARIANTS:
        path = f"{csv_prefix}_{v}.csv"
        df = pd.read_csv(path)
        df = compute_p_correct(df)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    parsed = df["image"].str.extract(r"phA(\d+)_phB(\d+)\.png$")
    df["phA"] = parsed[0].astype(int)
    df["phB"] = parsed[1].astype(int)

    grid_df = df.groupby(["phA", "phB"])["p_correct"].mean().reset_index()
    grid = np.full((16, 16), np.nan)
    for _, row in grid_df.iterrows():
        grid[int(row["phA"]), int(row["phB"])] = row["p_correct"]

    mean_acc = np.nanmean(grid)
    return grid, mean_acc


def load_cell_class(path):
    """Load cell_class.json and return consistent/counter masks as 16x16 bool arrays."""
    with open(path) as f:
        raw = json.load(f)
    con_mask = np.zeros((16, 16), dtype=bool)
    ctr_mask = np.zeros((16, 16), dtype=bool)
    for key, label in raw.items():
        phA, phB = map(int, key.split(","))
        if label == "consistent":
            con_mask[phA, phB] = True
        elif label == "counter":
            ctr_mask[phA, phB] = True
    return con_mask, ctr_mask


def plot_row(grids, labels, mask, suptitle, output, angles, major_angles, extent):
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6),
                              gridspec_kw={"wspace": 0.1},
                              constrained_layout=True)
    if n == 1:
        axes = [axes]

    for i, (ax, grid, label) in enumerate(zip(axes, grids, labels)):
        display = grid.copy()
        display[~mask] = np.nan

        gray_bg = np.full((16, 16), 0.8)
        gray_bg[mask] = np.nan
        ax.imshow(gray_bg, cmap="Greys", aspect="equal", vmin=0, vmax=1,
                  extent=extent, alpha=0.7)

        im = ax.imshow(display, cmap="seismic", aspect="equal", vmin=0, vmax=1,
                       extent=extent)

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("$\\theta_1$", fontsize=20)
        if i == 0:
            ax.set_ylabel("$\\theta_2$", fontsize=20)
        ax.set_xticks(major_angles)
        ax.set_yticks(major_angles)
        ax.set_xticklabels([f"{a:.0f}" for a in major_angles], fontsize=20)
        ax.set_yticklabels([f"{a:.0f}" for a in major_angles], fontsize=20)

        ax.text(0.5, -0.05, label, transform=ax.transAxes,
                fontsize=20, ha="center", va="top", fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, label="Accuracy", shrink=0.8, pad=0.02)
    cbar.set_label("Accuracy", fontsize=20)
    cbar.ax.tick_params(labelsize=13)

    if suptitle:
        fig.suptitle(suptitle, fontsize=30, fontweight="normal")
    plt.savefig(output, dpi=150, format="pdf")
    print(f"Saved {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", nargs="+", required=True,
                        help="CSV prefixes (e.g. logit_results_vqa_phase)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each model (e.g. 'base' '400k' '2m')")
    parser.add_argument("--cell-class", type=str, required=True,
                        help="Path to cell_class.json")
    parser.add_argument("--output", "-o", type=str, default="heatmap_compare.pdf")
    parser.add_argument("--suptitle", type=str, default="")
    args = parser.parse_args()

    assert len(args.prefixes) == len(args.labels)

    grids = []
    accs = []
    for prefix in args.prefixes:
        grid, acc = load_grid(prefix)
        grids.append(grid)
        accs.append(acc)

    con_mask, ctr_mask = load_cell_class(args.cell_class)

    angles = np.arange(16) * 22.5
    major_angles = np.arange(0, 360, 90)
    extent = [angles[0] - 11.25, angles[-1] + 11.25,
              angles[-1] + 11.25, angles[0] - 11.25]

    # Output paths: foo.png -> foo_consistent.png, foo_counter.png
    base = args.output.rsplit(".", 1)[0]
    ext = args.output.rsplit(".", 1)[1] if "." in args.output else "pdf"

    con_title = f"{args.suptitle} ($\\bf{{consistent}}$)" if args.suptitle else "$\\bf{consistent}$"
    ctr_title = f"{args.suptitle} ($\\bf{{counter}}$)" if args.suptitle else "$\\bf{counter}$"

    plot_row(grids, args.labels, con_mask, con_title,
             f"{base}_consistent.{ext}", angles, major_angles, extent)
    plot_row(grids, args.labels, ctr_mask, ctr_title,
             f"{base}_counter.{ext}", angles, major_angles, extent)


if __name__ == "__main__":
    main()
