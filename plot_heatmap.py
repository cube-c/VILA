import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output PNG file")
    parser.add_argument("--title", type=str, default="Yes - No Logit Difference (16x16)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["logit_diff"] = df["Yes_logit"] - df["No_logit"]

    # Sort by image name and reshape into 16x16
    df = df.sort_values("image").reset_index(drop=True)
    grid = df["logit_diff"].values.reshape(16, 16)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="RdBu_r", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, label="Logit diff (Yes - No)")

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
