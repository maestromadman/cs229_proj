"""b_sweep.py -- closed-class density x project_rounds sweep.

Extends the original 5-config experiment by varying *two* knobs:

  1. DET/PREP density. Same as experiment B (closed-class shrinkage),
     pushed progressively smaller to locate the floor.
  2. project_rounds. The parser's computational budget. The published
     parser uses 20; observed mean rounds-to-stabilize is ~5-10. Cutting
     the budget down to 5 stresses the smaller-k configs, which need
     more rounds to converge.

Every other parser knob is held at the published value: p=0.1
connectome sparsity, min_overlap=0.7 readout threshold, pre-allocated
LEX assemblies + Hebbian plasticity. Only DET, PREP density and
project_rounds vary across cells.

Grid: 5 density variants x 2 project_rounds values (20 = published, 5 =
stress) = 10 cells.

Usage
-----
  # Smoke test (1 sentence per template, single worker):
  python 03_density/b_sweep.py --variant baseline --rounds 20 \\
      --limit-per-template 1 --workers 1

  # Run a single cell of the grid:
  python 03_density/b_sweep.py --variant B_3 --rounds 5

  # Run the full 20-cell grid in one process (each cell gets its own Pool):
  python 03_density/b_sweep.py --all --workers 4

JSON output: one file per cell at
  03_density/results/b_sweep/b_sweep_{variant}_r{rounds}.json
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import defaultdict


# ---------------------------------------------------------------------
# Path setup -- must be before importing brain / parser / configs.
# ---------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
REPRO_DIR = os.path.join(REPO_ROOT, "02_parser_reproduced")
for _p in (REPO_ROOT, REPRO_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import brain as ac_brain  # noqa: E402
import numpy as np  # noqa: E402

# Reuse infrastructure from run_parser_experiments. Importing it also
# installs the Brain.__init__ seed patch exactly once per interpreter
# (parent + each spawn worker), so we do not install a second one here.
import run_parser_experiments as rpe  # noqa: E402
from run_parser_experiments import (  # noqa: E402
    _empty_aggregate,
    _absorb_result,
    _finalize_aggregate,
)

from configs import validate_config  # noqa: E402
from instrumented_parser import (  # noqa: E402
    ConfigurableEnglishParserBrain,
    parse_sentence_instrumented,
)
from test_sentences import SENTENCES, TEMPLATE_DESCRIPTIONS  # noqa: E402


# ---------------------------------------------------------------------
# The sweep grid.
# ---------------------------------------------------------------------

# DET/PREP (n, k). All other recurrent areas stay at (10000, 100).
# k/n = 0.01 holds for every entry.
DENSITY_VARIANTS = {
    "baseline":   (10000, 100),
    "B_original": (4000, 40),
    "B_2":        (2000, 20),
    "B_3":        (1000, 10),
    "B_4":        (500, 5),
}

# Round-budget settings to evaluate at. 20 is the published value
# (generous headroom). 5 is right at the floor of observed mean
# rounds-to-stabilize from the prior experiment. 10 is the intermediate
# point added after the initial run to fill in the accuracy curve
# between the headroom regime and the stress regime.
PROJECT_ROUNDS_VALUES = [20, 10, 5]


def build_config(det_prep_n, det_prep_k):
    """Build a per-area config dict for one density variant.

    Only DET and PREP take the variant's (n, k); every other recurrent
    area is held at the published baseline 10000/100.
    """
    return {
        "LEX":   {"n": 10000, "k": 100},
        "VERB":  {"n": 10000, "k": 100},
        "SUBJ":  {"n": 10000, "k": 100},
        "OBJ":   {"n": 10000, "k": 100},
        "DET":   {"n": det_prep_n, "k": det_prep_k},
        "ADJ":   {"n": 10000, "k": 100},
        "ADV":   {"n": 10000, "k": 100},
        "PREP":  {"n": det_prep_n, "k": det_prep_k},
        "PREPP": {"n": 10000, "k": 100},
    }


# ---------------------------------------------------------------------
# Per-sentence driver with configurable project_rounds.
# ---------------------------------------------------------------------

def parse_one(area_config, sentence, project_rounds):
    """Fresh brain, parse one sentence, discard. Returns a result dict.

    Same as run_parser_experiments.parse_one but with project_rounds
    plumbed through to parse_sentence_instrumented.
    """
    brain = None
    try:
        brain = ConfigurableEnglishParserBrain(
            p=0.1, area_config=area_config, verbose=False,
        )
        deps, conv = parse_sentence_instrumented(
            brain, sentence, project_rounds=project_rounds,
        )
        return {"dependencies": deps, "convergence": conv, "error": None}
    except Exception as e:  # noqa: BLE001
        return {
            "dependencies": None,
            "convergence": None,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    finally:
        del brain
        gc.collect()


# ---------------------------------------------------------------------
# Multiprocessing scaffolding.
# ---------------------------------------------------------------------

def _worker_init(seed):
    """Each spawn worker re-imports this module (which re-imports
    run_parser_experiments and reinstalls the Brain seed patch).
    Then we point that patch's seed slot at the right value."""
    rpe._CURRENT_SEED[0] = seed


def _worker_parse(args):
    """Top-level so it pickles cleanly. One sentence per call."""
    sentence_idx, area_config, sentence_info, project_rounds = args
    r = parse_one(area_config, sentence_info["sentence"], project_rounds)
    return sentence_idx, sentence_info, r


# ---------------------------------------------------------------------
# Aggregation for one grid cell.
# ---------------------------------------------------------------------

def aggregate_cell(variant_name, area_config, sentences, project_rounds,
                   *, workers=1, seed=0):
    """Run all sentences under one (variant, project_rounds) cell.

    Uses the same _empty_aggregate / _absorb_result / _finalize_aggregate
    helpers as run_parser_experiments so the output schema matches.
    """
    agg = _empty_aggregate()
    t0 = time.time()
    label = f"{variant_name}_r{project_rounds}"

    if workers <= 1:
        for i, sinfo in enumerate(sentences):
            r = parse_one(area_config, sinfo["sentence"], project_rounds)
            _absorb_result(agg, sinfo, r)
            if (i + 1) % 20 == 0 or (i + 1) == len(sentences):
                elapsed = time.time() - t0
                print(
                    f"  [{label}] {i + 1}/{len(sentences)} "
                    f"correct={agg['n_correct']} elapsed={elapsed:.0f}s",
                    flush=True,
                )
    else:
        ctx = mp.get_context("spawn")
        n = len(sentences)
        printed = 0
        worker_args = [
            (i, area_config, s, project_rounds)
            for i, s in enumerate(sentences)
        ]
        with ctx.Pool(
            workers,
            initializer=_worker_init,
            initargs=(seed,),
        ) as pool:
            for sentence_idx, sinfo, r in pool.imap_unordered(
                _worker_parse, worker_args, chunksize=1,
            ):
                _absorb_result(agg, sinfo, r)
                printed += 1
                if printed % 20 == 0 or printed == n:
                    elapsed = time.time() - t0
                    print(
                        f"  [{label}] {printed}/{n} "
                        f"correct={agg['n_correct']} "
                        f"elapsed={elapsed:.0f}s ({workers} workers)",
                        flush=True,
                    )

    elapsed = time.time() - t0
    result = _finalize_aggregate(variant_name, area_config, agg, elapsed)
    # Tag with sweep coordinates so downstream analysis is self-describing.
    result["variant"] = variant_name
    result["project_rounds"] = project_rounds
    result["density"] = list(DENSITY_VARIANTS[variant_name])
    return result


# ---------------------------------------------------------------------
# Sentence selection (full 18 active templates, optional cap for smoke).
# ---------------------------------------------------------------------

def filter_sentences(limit_per_template=None):
    active_tids = {tid for tid in TEMPLATE_DESCRIPTIONS if tid not in (14, 20)}
    if limit_per_template is None:
        return [s for s in SENTENCES if s["template_id"] in active_tids]
    out = []
    per_tid = defaultdict(int)
    for s in SENTENCES:
        tid = s["template_id"]
        if tid in active_tids and per_tid[tid] < limit_per_template:
            out.append(s)
            per_tid[tid] += 1
    return out


# ---------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------

def save_result(out_dir, result):
    label = f"{result['variant']}_r{result['project_rounds']}"
    path = os.path.join(out_dir, f"b_sweep_{label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {path}", flush=True)
    return path


def print_grid(all_results):
    """Print the 2D grid: density variants x project_rounds values."""
    variants = list(DENSITY_VARIANTS.keys())
    rounds = PROJECT_ROUNDS_VALUES

    width = 78
    print("\n" + "=" * width)
    print("B-SWEEP GRID")
    print("=" * width)

    header = f"{'variant':<13}{'(n, k)':<14}" + "".join(
        f"  r={r:<3} " for r in rounds
    )

    # Accuracy.
    print("\nAccuracy:")
    print(header)
    print("-" * len(header))
    for v in variants:
        n, k = DENSITY_VARIANTS[v]
        row = f"{v:<13}({n:>4},{k:>3})    "
        for r in rounds:
            ov = all_results.get((v, r), {}).get("overall", {})
            acc = ov.get("accuracy")
            cell = f"{acc * 100:5.1f}%" if acc is not None else "   -  "
            row += f"{cell:<7} "
        print(row)

    # Mean rounds per word.
    print("\nMean rounds per word:")
    print(header)
    print("-" * len(header))
    for v in variants:
        n, k = DENSITY_VARIANTS[v]
        row = f"{v:<13}({n:>4},{k:>3})    "
        for r in rounds:
            ov = all_results.get((v, r), {}).get("overall", {})
            rpw = ov.get("mean_rounds_per_word")
            cell = f"{rpw:5.2f}" if rpw is not None else "  -  "
            row += f"{cell:<7} "
        print(row)


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="B-extension sweep: DET/PREP density x project_rounds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variant",
        choices=list(DENSITY_VARIANTS),
        help="Single density variant. Required unless --all is given.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        choices=PROJECT_ROUNDS_VALUES,
        help="Single project_rounds value. Required unless --all is given.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all 5 density variants x 2 round settings = 10 cells.",
    )
    p.add_argument("--seed", type=int, default=0)
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
        "--out-dir", default=None,
        help="Override output dir (default: 03_density/results/b_sweep/).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # PYTHONHASHSEED has to be set in the env before Python starts using
    # string hashing. Re-exec on the parent only; spawn workers inherit
    # PYTHONHASHSEED from the parent process and so do not re-exec.
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    rpe._CURRENT_SEED[0] = args.seed

    # Which cells to run?
    if args.all:
        cells = [
            (v, r) for v in DENSITY_VARIANTS for r in PROJECT_ROUNDS_VALUES
        ]
    else:
        if args.variant is None or args.rounds is None:
            print(
                "Either pass --all, or pass both --variant and --rounds.",
                file=sys.stderr,
            )
            sys.exit(2)
        cells = [(args.variant, args.rounds)]

    out_dir = args.out_dir or os.path.join(HERE, "results", "b_sweep")
    os.makedirs(out_dir, exist_ok=True)

    sentences = filter_sentences(limit_per_template=args.limit_per_template)

    print(f"B-sweep: {len(cells)} cell(s), {len(sentences)} sentences each")
    print(
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}; "
        f"seed={args.seed}; workers={args.workers}"
    )
    print(f"Output dir: {out_dir}")
    if args.limit_per_template is not None:
        print(f"limit_per_template = {args.limit_per_template}")

    all_results = {}
    for variant, rounds in cells:
        n, k = DENSITY_VARIANTS[variant]
        cfg = build_config(n, k)
        # Will raise loudly if k/n != 0.01 or any required area missing.
        validate_config(f"b_sweep:{variant}_r{rounds}", cfg)

        print(f"\n--- {variant} (DET/PREP={n}/{k}), project_rounds={rounds} ---")
        result = aggregate_cell(
            variant, cfg, sentences, rounds,
            workers=args.workers, seed=args.seed,
        )
        all_results[(variant, rounds)] = result
        save_result(out_dir, result)

    if len(cells) > 1:
        print_grid(all_results)


if __name__ == "__main__":
    main()
