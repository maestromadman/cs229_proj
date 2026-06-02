"""b_sweep_reseed.py -- reseed the 5 frontier-adjacent cells with N=10 seeds each.

For each of the following 5 (variant, project_rounds) cells, run the full
180-sentence sweep at 10 different RNG seeds (0..9), then compute mean and
standard deviation of accuracy and mean-rounds-per-word across seeds.

  Cell                Why
  -----               ---
  B_2 / r=10          Threshold lower bound (single-seed: 63.9%)
  B_2 / r=11          Threshold itself (single-seed: 100%)
  B_2 / r=12          Threshold upper-bound check (single-seed: 100%)
  B_3 / r=10          Near-cliff bump (single-seed: 42.8% vs 40.6% floor)
  B_original / r=10   Control: seed variance in converged regime (s-seed: 100%)

Determinism note
----------------
We pin PYTHONHASHSEED=0 once at the parent (re-exec if needed) and then
vary only the numpy RNG seed across runs. Python's hash seed governs dict
iteration order inside the parser; the numpy seed governs connectome
generation. The connectome is the dominant variance source we want to
characterize, so this scope is intentional. A stricter rerun (one
subprocess per seed, each with its own PYTHONHASHSEED) is a follow-up
if dict-order noise turns out to matter.

Usage
-----
  # Smoke test (1 sentence per template, single worker, single cell):
  python 03_density/b_sweep_reseed.py --limit-per-template 1 --workers 1 \\
      --seeds 0 --only B_2 11

  # Full sweep (5 cells x 10 seeds = 50 runs):
  python 03_density/b_sweep_reseed.py --workers 4

  # Regenerate figure from already-run JSONs (no parsing):
  python 03_density/b_sweep_reseed.py --figure-only

JSON output: one file per (cell, seed) at
  03_density/results/b_sweep/b_sweep_reseed_{variant}_r{rounds}_s{seed}.json

Figure output: 03_density/results/figure_3_seeded.png
"""

import argparse
import json
import os
import sys
import time


# ---------------------------------------------------------------------
# Path setup -- must be before importing brain / parser / configs.
# ---------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
REPRO_DIR = os.path.join(REPO_ROOT, "02_parser_reproduced")
for _p in (REPO_ROOT, REPRO_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------
# PYTHONHASHSEED pin (re-exec parent once if needed).
# ---------------------------------------------------------------------

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)


import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_parser_experiments as rpe  # noqa: E402
from b_sweep import (  # noqa: E402
    DENSITY_VARIANTS, build_config, aggregate_cell, filter_sentences,
)
from configs import validate_config  # noqa: E402


# ---------------------------------------------------------------------
# The 5 cells to reseed.
# ---------------------------------------------------------------------

CELLS_TO_RESEED = [
    ("B_2", 10),
    ("B_2", 11),
    ("B_2", 12),
    ("B_3", 10),
    ("B_original", 10),
]

# Single-seed reference values from the existing b_sweep JSONs (for display
# in the figure caption alongside the reseeded mean +/- std).
SINGLE_SEED_REFERENCE = {
    ("B_2", 10): 0.6389,
    ("B_2", 11): 1.0,
    ("B_2", 12): 1.0,
    ("B_3", 10): 0.4278,
    ("B_original", 10): 1.0,
}

DEFAULT_SEEDS = list(range(10))

RESULTS_BASE = os.path.join(HERE, "results")
OUT_DIR = os.path.join(RESULTS_BASE, "b_sweep")


# ---------------------------------------------------------------------
# Single-run driver.
# ---------------------------------------------------------------------

def reseed_path(variant, rounds, seed):
    return os.path.join(
        OUT_DIR, f"b_sweep_reseed_{variant}_r{rounds}_s{seed}.json"
    )


def run_one(variant, rounds, seed, sentences, workers):
    """One (cell, seed) run. Writes its JSON; returns the result dict."""
    rpe._CURRENT_SEED[0] = seed
    n, k = DENSITY_VARIANTS[variant]
    cfg = build_config(n, k)
    validate_config(f"reseed:{variant}_r{rounds}_s{seed}", cfg)

    result = aggregate_cell(
        variant, cfg, sentences, rounds,
        workers=workers, seed=seed,
    )
    result["seed"] = seed
    out_path = reseed_path(variant, rounds, seed)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------
# Aggregation across seeds.
# ---------------------------------------------------------------------

def summarize(all_results):
    """all_results: dict[(variant, rounds)] -> list[result dict]."""
    summary = {}
    for cell, results in all_results.items():
        if not results:
            continue
        accs = [r["overall"]["accuracy"] for r in results]
        rpws = [r["overall"]["mean_rounds_per_word"] for r in results]
        seeds = [r.get("seed", -1) for r in results]
        summary[cell] = {
            "n_seeds": len(results),
            "seeds": seeds,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "acc_min": float(min(accs)),
            "acc_max": float(max(accs)),
            "rpw_mean": float(np.mean(rpws)),
            "rpw_std": float(np.std(rpws, ddof=1)) if len(rpws) > 1 else 0.0,
            "per_seed_accs": accs,
            "per_seed_rpws": rpws,
        }
    return summary


def load_results(seeds=None):
    """Reload per-(cell, seed) JSONs from disk for figure aggregation.

    If ``seeds`` is None (default), pulls every JSON that exists on disk
    for each cell, sorted by seed. This lets a partial run (some cells
    with N seeds, others with M) all contribute to the figure.

    If ``seeds`` is a list, only loads those specific seeds.
    """
    import glob
    all_results = {}
    for variant, rounds in CELLS_TO_RESEED:
        runs = []
        if seeds is None:
            pattern = os.path.join(
                OUT_DIR, f"b_sweep_reseed_{variant}_r{rounds}_s*.json"
            )
            paths = sorted(glob.glob(pattern), key=lambda p: int(
                p.rsplit("_s", 1)[1].split(".json")[0]
            ))
        else:
            paths = [reseed_path(variant, rounds, s) for s in seeds]
            paths = [p for p in paths if os.path.exists(p)]
        for path in paths:
            with open(path) as f:
                runs.append(json.load(f))
        if runs:
            all_results[(variant, rounds)] = runs
    return all_results


# ---------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------

def make_figure(summary, out_path):
    """Two-panel figure: per-cell accuracy bar chart + summary table."""
    if not summary:
        print("No results to plot; skipping figure.")
        return

    cells = [c for c in CELLS_TO_RESEED if c in summary]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.45)

    # Top: bar chart of mean accuracy with +/- 1 SD error bars. Cells
    # with zero std (the 4 deterministic ones) just have no visible
    # error bar; B_3 r=10 is the only cell where the error bar is
    # visible, which makes the "one of these is not like the others"
    # story immediately readable.
    ax = fig.add_subplot(gs[0])
    xs = list(range(len(cells)))
    labels = []
    for v, r in cells:
        n, k = DENSITY_VARIANTS[v]
        labels.append(f"{v}\n(n={n}, k={k})\nr={r}")

    means = [summary[c]["acc_mean"] for c in cells]
    stds = [summary[c]["acc_std"] for c in cells]

    # Color B_3 differently to draw attention to the only cell with
    # real seed variance.
    colors = []
    for cell in cells:
        if cell == ("B_3", 10):
            colors.append("#d96b6b")  # muted red
        else:
            colors.append("#5b8dbf")  # steel blue

    ax.bar(
        xs, means, yerr=stds, capsize=10,
        color=colors, edgecolor="black", linewidth=0.8,
        ecolor="black", error_kw={"elinewidth": 1.8},
    )

    # Annotate each bar with the value above it.
    for i, (m, s) in enumerate(zip(means, stds)):
        if s > 0:
            label = f"{m * 100:.1f}% ± {s * 100:.1f}pp"
        else:
            label = f"{m * 100:.1f}%"
        y_text = m + max(s + 0.015, 0.025)
        ax.text(i, y_text, label, ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (180 sentences per seed)")
    ax.set_ylim(0, 1.18)
    n_seeds = summary[cells[0]]["n_seeds"] if cells else 0
    ax.set_title(
        f"Mean accuracy across {n_seeds} seeds "
        "(error bars = +/- 1 SD; only B_3 r=10 shows real seed variance)"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Bottom: summary table.
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")
    headers = [
        "Cell", "(n_DP, k_DP)", "rounds", "n seeds",
        "Acc mean", "Acc std", "Acc min", "Acc max",
        "RPW mean", "RPW std",
    ]
    rows = []
    for cell in cells:
        v, r = cell
        n, k = DENSITY_VARIANTS[v]
        s = summary[cell]
        rows.append([
            v,
            f"({n}, {k})",
            str(r),
            str(s["n_seeds"]),
            f"{s['acc_mean'] * 100:.2f}%",
            f"{s['acc_std'] * 100:.2f}",
            f"{s['acc_min'] * 100:.2f}%",
            f"{s['acc_max'] * 100:.2f}%",
            f"{s['rpw_mean']:.3f}",
            f"{s['rpw_std']:.3f}",
        ])
    table = ax_table.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    # Header styling.
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#dddddd")
        cell.set_text_props(weight="bold")

    fig.suptitle(
        "Figure 3: Seed-level confidence at frontier-adjacent cells "
        f"({n_seeds} seeds, 180 sentences each)",
        fontsize=12, y=0.98,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--limit-per-template", type=int, default=None,
        help="Cap sentences per template (for fast smoke tests).",
    )
    default_workers = min(os.cpu_count() or 1, 4)
    p.add_argument(
        "--workers", type=int, default=default_workers,
        help=f"Parallel sentence workers (default {default_workers}).",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Seeds to run (default {DEFAULT_SEEDS}).",
    )
    p.add_argument(
        "--only", nargs=2, metavar=("VARIANT", "ROUNDS"), default=None,
        help="Run only one cell (e.g. --only B_2 11). Useful for smoke test.",
    )
    p.add_argument(
        "--figure-only", action="store_true",
        help="Skip runs; load existing JSONs and regenerate the figure.",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="For each (cell, seed) tuple, skip the run if its JSON "
             "already exists on disk. Useful for resuming an interrupted "
             "sweep without re-running completed cells.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    sentences = filter_sentences(limit_per_template=args.limit_per_template)

    if args.only is not None:
        variant = args.only[0]
        rounds = int(args.only[1])
        cells = [(variant, rounds)]
    else:
        cells = CELLS_TO_RESEED

    # Build the per-(cell, seed) execution list, honoring --skip-existing.
    to_run = []
    skipped = []
    for variant, rounds in cells:
        for seed in args.seeds:
            path = reseed_path(variant, rounds, seed)
            if args.skip_existing and os.path.exists(path):
                skipped.append((variant, rounds, seed))
                continue
            to_run.append((variant, rounds, seed))

    n_total = len(to_run)
    print(f"Reseed sweep: {n_total} runs to execute "
          f"({len(skipped)} skipped via --skip-existing)")
    print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}; "
          f"workers={args.workers}; sentences/cell={len(sentences)}")
    print(f"Output dir: {OUT_DIR}")
    if skipped:
        for v, r, s in skipped:
            print(f"  SKIP (exists): {v} r={r} seed={s}")

    if args.figure_only:
        print("--figure-only: no runs executed")
    else:
        t_start = time.time()
        for run_idx, (variant, rounds, seed) in enumerate(to_run, 1):
            print(
                f"\n[{run_idx}/{n_total}] {variant} r={rounds} seed={seed}",
                flush=True,
            )
            t0 = time.time()
            r = run_one(variant, rounds, seed, sentences, args.workers)
            elapsed = time.time() - t0
            acc = r["overall"]["accuracy"]
            rpw = r["overall"]["mean_rounds_per_word"]
            print(
                f"  -> acc={acc * 100:.1f}%  rpw={rpw:.2f}  "
                f"({elapsed:.0f}s)",
                flush=True,
            )
        total_elapsed = time.time() - t_start
        print(f"\nAll {n_total} runs completed in "
              f"{total_elapsed / 60:.1f} min")

    # Reload JSONs from disk for the figure, restricted to the seeds
    # requested via --seeds. Cells whose seed range is a strict subset of
    # what's on disk will be aggregated over only the requested seeds,
    # which keeps the figure visually consistent (same n_seeds per cell).
    all_results = load_results(seeds=args.seeds)
    n_loaded = sum(len(v) for v in all_results.values())
    print(f"Loaded {n_loaded} per-(cell, seed) JSONs for the figure.")

    summary = summarize(all_results)
    fig_path = os.path.join(RESULTS_BASE, "figure_3_seeded.png")
    make_figure(summary, fig_path)


if __name__ == "__main__":
    main()
