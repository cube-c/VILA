import matplotlib.pyplot as plt
import numpy as np

VARIANT_TO_NUM = {"vanilla": 0, "80k": 80_000, "400k": 400_000, "800k": 800_000, "2M": 2_000_000, "RoboRefer": None}

DATA = {
    "NVILA-Lite-2B": {
        "variants": ["vanilla", "80k", "400k", "800k", "2M", "RoboRefer"],
        "v_bar":    [0.488, 0.499, 0.669, 0.646, 0.812, 0.793],
        "acc_cons": [0.504, 0.562, 0.804, 0.728, 0.875, 0.816],
        "acc_ctr":  [0.471, 0.438, 0.538, 0.571, 0.749, 0.770],
    },
    "Molmo-7B": {
        "variants": ["vanilla", "80k", "400k", "800k", "2M"],
        "v_bar":    [0.528, 0.496, 0.501, 0.531, 0.666],
        "acc_cons": [0.565, 0.507, 0.593, 0.628, 0.703],
        "acc_ctr":  [0.487, 0.486, 0.409, 0.430, 0.630],
    },
    "Qwen2.5-VL-3B": {
        "variants": ["vanilla", "80k", "400k", "800k", "2M"],
        "v_bar":    [0.570, 0.512, 0.503, 0.499, 0.500],
        "acc_cons": [0.776, 0.585, 0.588, 0.600, 0.648],
        "acc_ctr":  [0.360, 0.440, 0.418, 0.398, 0.353],
    },
}

for model, d in DATA.items():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Split numeric variants from non-numeric (e.g. RoboRefer)
    x_nums = []
    labels = []
    extra = {}  # non-numeric variants plotted separately
    for i, v in enumerate(d["variants"]):
        num = VARIANT_TO_NUM[v]
        if num is not None:
            x_nums.append(num)
            labels.append(v)
        else:
            extra[v] = i

    x = np.array(x_nums)
    idx_num = [i for i, v in enumerate(d["variants"]) if VARIANT_TO_NUM[v] is not None]

    ax.plot(x, [d["v_bar"][i] for i in idx_num],    "o-", markersize=4, label=r"$\bar{v}$ (mean P(correct))", color="tab:blue")
    ax.plot(x, [d["acc_cons"][i] for i in idx_num], "s-", markersize=4, label=r"Acc$_{\mathrm{cons}}$ (consistent)", color="tab:green")
    ax.plot(x, [d["acc_ctr"][i] for i in idx_num],  "^-", markersize=4, label=r"Acc$_{\mathrm{ctr}}$ (counter)", color="tab:red")

    # Plot non-numeric variants as standalone markers at the right edge
    for v, i in extra.items():
        x_extra = x[-1] * 1.15
        ax.plot(x_extra, d["v_bar"][i],    "o", color="tab:blue", markersize=5)
        ax.plot(x_extra, d["acc_cons"][i], "s", color="tab:green", markersize=5)
        ax.plot(x_extra, d["acc_ctr"][i],  "^", color="tab:red", markersize=5)
        ax.axvline(x_extra, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        labels.append(v)
        x = np.append(x, x_extra)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Training data size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.2, 1.0)
    ax.set_title(model)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = f"accuracy_lines_{model.replace(' ', '_').replace('.', '_').replace('-', '_')}.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)
