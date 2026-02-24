import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output PNG file")
    parser.add_argument("--title", type=str, default="Yes - No Logit Difference (16x16)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["logit_diff"] = df["Yes_logit"] - df["No_logit"]
    df["p_yes"] = 1.0 / (1.0 + np.exp(-df["logit_diff"]))

    # Sort by image name and reshape into 16x16
    df = df.sort_values("image").reset_index(drop=True)
    grid = df["p_yes"].values.reshape(16, 16)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="seismic", aspect="equal", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, label="P(Yes)")

    # Compute RMSE against ground truth if available
    if "ground_truth" in df.columns:
        gt_binary = (df["ground_truth"] == "Yes").astype(float)
        rmse = np.sqrt(np.mean((df["p_yes"] - gt_binary) ** 2))
        ax.text(0.02, 0.98, f"RMSE = {rmse:.4f}", transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Blue Idx")
    ax.set_ylabel("Red Idx")
    ax.set_title(args.title)
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
