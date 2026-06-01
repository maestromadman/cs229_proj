"""make_figures.py -- publication figures for the b_sweep results.

Reads JSON output from 03_density/results/b_sweep/ and writes:

  03_density/results/figure_1_accuracy_heatmap.png
      A 5-row x 2-column heatmap of accuracy (DET/PREP density rows,
      project_rounds columns). Annotated with the percentage and the raw
      count per cell. Diverging RdYlGn colormap so the "headroom"
      (green) vs "below floor" (red) regimes are visually distinct.

  03_density/results/figure_2_density_cliff.png
      Line plot of accuracy vs DET/PREP cap size k (log x-axis), one
      line per project_rounds value (color/marker coded). Shaded band
      highlights the cliff region between k=20 and k=10.

All figures are 300 DPI PNG, suitable for an Overleaf paper.

Usage:
  python 03_density/make_figures.py
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Paths and ordering.
# ---------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "results", "b_sweep")
OUT_DIR = os.path.join(HERE, "results")

# Top-to-bottom in the heatmap, left-to-right in the line plot.
VARIANTS = ["baseline", "B_original", "B_2", "B_3", "B_4"]
ROUNDS = [20, 5]

# Color per round budget (consistent across figures so the reader can
# track which line is which).
ROUND_COLORS = {
    20: "#1f77b4",  # tab:blue
    5:  "#d62728",  # tab:red
}
ROUND_MARKERS = {
    20: "o",
    5:  "s",
}


# ---------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------

def load_sweep_data():
    """Return dict keyed by (variant, rounds) -> result dict."""
    data = {}
    for f in sorted(glob.glob(os.path.join(SWEEP_DIR, "*.json"))):
        d = json.load(open(f))
        data[(d["variant"], d["project_rounds"])] = d
    return data


def warn_if_missing(data):
    expected = {(v, r) for v in VARIANTS for r in ROUNDS}
    missing = sorted(expected - set(data.keys()))
    if missing:
        print(
            f"WARNING: missing {len(missing)} cells; figures may be incomplete:\n"
            f"  {missing}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------
# Figure 1: accuracy heatmap.
# ---------------------------------------------------------------------

def figure_1_heatmap(data, outpath):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Build the accuracy matrix.
    acc = np.zeros((len(VARIANTS), len(ROUNDS)))
    counts = np.zeros_like(acc, dtype=int)
    for i, v in enumerate(VARIANTS):
        for j, r in enumerate(ROUNDS):
            cell = data.get((v, r))
            if cell is None:
                acc[i, j] = np.nan
                counts[i, j] = -1
            else:
                acc[i, j] = cell["overall"]["accuracy"] * 100
                counts[i, j] = cell["overall"]["total_correct"]

    # Single-hue blue colormap. Light = low accuracy, dark = high.
    im = ax.imshow(acc, cmap="Blues", aspect="auto", vmin=0, vmax=100)

    # Annotate every cell with percentage + raw count.
    for i in range(len(VARIANTS)):
        for j in range(len(ROUNDS)):
            if np.isnan(acc[i, j]):
                ax.text(j, i, "missing", ha="center", va="center",
                        color="gray", fontsize=10, style="italic")
                continue
            # On a Blues colormap, dark cells (~80%+) need white text;
            # light cells need black.
            txt_color = "white" if acc[i, j] >= 60 else "black"
            ax.text(
                j, i,
                f"{acc[i, j]:.1f}%\n({counts[i, j]}/180)",
                ha="center", va="center",
                color=txt_color, fontsize=11, fontweight="bold",
            )

    # y-axis labels: variant name + (n, k).
    y_labels = []
    for v in VARIANTS:
        cell = data.get((v, ROUNDS[0])) or data.get((v, ROUNDS[1]))
        if cell is None:
            y_labels.append(f"{v}\n(?, ?)")
        else:
            n, k = cell["density"]
            y_labels.append(f"{v}\n(n={n}, k={k})")
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels(y_labels, fontsize=10)

    # x-axis labels: just the number, with the meaning in the axis label.
    ax.set_xticks(range(len(ROUNDS)))
    ax.set_xticklabels([str(r) for r in ROUNDS], fontsize=12)
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)

    ax.set_xlabel("# of Projection Rounds", fontsize=12, labelpad=10)
    ax.set_ylabel(
        "DET/PREP area size (n, k); sparsity k/n = 0.01",
        fontsize=11, labelpad=10,
    )
    # No title — the colorbar already labels the heatmap values as
    # "Accuracy (%)", and the methodology fits more naturally in the
    # paper caption than in the figure itself.

    # Colorbar.
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label("Accuracy (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------
# Figure 2: accuracy vs k line plot, with cliff annotation.
# ---------------------------------------------------------------------

def figure_2_cliff(data, outpath):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # k values in the order of VARIANTS.
    k_values = []
    for v in VARIANTS:
        cell = data.get((v, ROUNDS[0])) or data.get((v, ROUNDS[1]))
        k_values.append(cell["density"][1] if cell else np.nan)

    for r in ROUNDS:
        accs = []
        for v in VARIANTS:
            cell = data.get((v, r))
            accs.append(cell["overall"]["accuracy"] * 100 if cell else np.nan)
        ax.plot(
            k_values, accs,
            marker=ROUND_MARKERS[r], markersize=9,
            linewidth=2.2,
            color=ROUND_COLORS[r],
            label=f"project_rounds = {r}",
            zorder=3,
        )

    # Shade the BELOW-FLOOR region (k <= 10), where the parser hits
    # the capacity-limited plateau and additional compute does not
    # rescue it. The cliff in the r=20 line happens at the right edge
    # of this band (between k=10 and k=20); we leave the cliff edge
    # unshaded so the visual story is "everything in the shaded zone
    # is broken, everything to the right is above the floor."
    ax.axvspan(4.5, 10, alpha=0.13, color="#cc4444", zorder=1)
    ax.text(
        np.sqrt(4.5 * 10), 6,  # geometric midpoint on log axis
        "below capacity floor",
        ha="center", va="bottom",
        fontsize=10, style="italic", color="#992222",
    )
    # Mark the cliff edge with a thin vertical line.
    ax.axvline(10, color="#992222", linestyle="--", linewidth=1,
               alpha=0.6, zorder=2)

    # 100% reference line.
    ax.axhline(100, color="gray", linestyle=":", linewidth=1, alpha=0.6,
               zorder=1)

    # Axis formatting.
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.minorticks_off()
    ax.set_xlabel(
        "DET/PREP cap size $k$ (log scale; population $n = 100k$; sparsity $k/n = 0.01$)",
        fontsize=11,
    )
    ax.set_ylabel("Parser accuracy (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(alpha=0.25, zorder=0)

    ax.legend(fontsize=11, loc="lower right", framealpha=0.95)
    ax.set_title(
        "Capacity floor at $k = 10$: below-floor accuracy is "
        "independent of projection rounds",
        fontsize=12, pad=10,
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------

def main():
    data = load_sweep_data()
    warn_if_missing(data)
    os.makedirs(OUT_DIR, exist_ok=True)

    figure_1_heatmap(
        data,
        os.path.join(OUT_DIR, "figure_1_accuracy_heatmap.png"),
    )
    figure_2_cliff(
        data,
        os.path.join(OUT_DIR, "figure_2_density_cliff.png"),
    )


if __name__ == "__main__":
    main()
