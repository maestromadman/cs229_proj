"""Heterogeneous Neuron Density experiments on the Assembly-Calculus parser.

Runs 5 per-area-config variants of the parser
(``baseline``, ``experiment_A_verb_enlarged``,
``experiment_B_closed_class_shrunk``, ``experiment_C_proportional``,
``experiment_D_inverted``) on:

  Round 1: 6 priority templates (10 sentences each) =  60 sents/config.
  Full:    all 18 working templates                  = 180 sents/config.

Per-config JSON dropped to ``03_density/results/{name}_{round1,full}.json``.

Usage
-----
  # All configs, both rounds, with the Round 1 summary printed first:
  python 03_density/run_parser_experiments.py

  # Single config:
  python 03_density/run_parser_experiments.py --config experiment_A_verb_enlarged

  # Round 1 only:
  python 03_density/run_parser_experiments.py --round1-only

  # Full only (skip Round 1 summary):
  python 03_density/run_parser_experiments.py --full-only

  # Parallel across sentences (default min(cpu_count, 8)):
  python 03_density/run_parser_experiments.py --workers 8

  # Smoke test:
  python 03_density/run_parser_experiments.py --limit-per-template 1 --workers 1

Determinism
-----------
The repo has two sources of randomness: ``brain.Brain``'s NumPy RNG, and
Python's per-process string hash seed (which affects ``defaultdict(set)``
iteration order inside the parser). We pin ``PYTHONHASHSEED`` via re-exec
on the parent, monkey-patch ``brain.Brain.__init__`` so every brain uses
``--seed``, and propagate the same seed to multiprocessing workers via a
Pool initializer. Workers inherit PYTHONHASHSEED from the parent
environment, so they do not re-exec.

Multiprocessing
---------------
Sentences are independent, so we parallelize across them with
``multiprocessing.Pool``. Each worker draws a fresh
``ConfigurableEnglishParserBrain`` per sentence (memory hits a peak of
~400 MB per worker while the LEX inner connectome materializes). With
N workers, peak RAM ~= 400 MB * N.
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
import traceback
from collections import defaultdict


# ---------------------------------------------------------------------
# Path setup (must happen before importing brain/parser modules).
# ---------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
REPRO_DIR = os.path.join(REPO_ROOT, "02_parser_reproduced")
for _p in (REPO_ROOT, REPRO_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import brain as ac_brain  # noqa: E402
import numpy as np  # noqa: E402

from configs import (  # noqa: E402
    CONFIGS, SHORT_NAMES, validate_config, validate_all,
)
from instrumented_parser import (  # noqa: E402
    ConfigurableEnglishParserBrain,
    parse_sentence_instrumented,
)
from test_sentences import (  # noqa: E402  (lives in 02_parser_reproduced)
    SENTENCES, TEMPLATE_DESCRIPTIONS,
)


# ---------------------------------------------------------------------
# Brain seed patch.
# ---------------------------------------------------------------------
#
# brain.Brain.__init__ already accepts a ``seed`` kwarg, but parser.py
# never sets it -- so by default every brain across the experiment uses
# the NumPy default RNG, which is non-reproducible across runs. We
# monkey-patch __init__ to substitute our seed.
#
# ``_CURRENT_SEED`` is module-level so it is visible to both the
# patched function and (after the Pool initializer sets it) worker
# processes. With ``spawn`` start method, workers re-import this
# module, _install_seed_patch() runs again, and ``_worker_init`` below
# overwrites _CURRENT_SEED[0] with the seed the parent passes in.

_CURRENT_SEED = [0]


def _install_seed_patch():
    """Patch ``Brain.__init__`` to seed *both* the brain's own RNG and
    NumPy's legacy global RNG.

    The legacy global is the RNG that ``scipy.stats.truncnorm.rvs`` and
    ``scipy.stats.binom`` fall back to when ``random_state`` is not
    passed. ``brain.project_into`` calls ``truncnorm.rvs(...)`` without
    a random_state (see brain.py: the ``potential_new_winner_inputs``
    line), so without this extra ``np.random.seed`` two runs at the
    same ``Brain(seed=0)`` produce different per-area dynamics.
    """
    _orig = ac_brain.Brain.__init__

    def _seeded_init(self, p, *args, **kwargs):
        kwargs.pop("seed", None)
        np.random.seed(_CURRENT_SEED[0])
        _orig(self, p, *args, seed=_CURRENT_SEED[0], **kwargs)

    ac_brain.Brain.__init__ = _seeded_init


_install_seed_patch()


# ---------------------------------------------------------------------
# Template metadata.
# ---------------------------------------------------------------------

ROUND1_TEMPLATES = [2, 5, 7, 13, 15, 16]

COMPLEXITY = {
    1: "simple", 2: "simple", 3: "simple", 4: "simple", 8: "simple",
    17: "simple", 18: "simple",
    5: "medium", 6: "medium", 7: "medium",
    9: "medium", 10: "medium", 11: "medium", 19: "medium",
    12: "complex", 13: "complex", 15: "complex", 16: "complex",
}


def _filter_sentences(template_ids, limit_per_template=None):
    out = []
    if limit_per_template is None:
        for s in SENTENCES:
            if s["template_id"] in template_ids:
                out.append(s)
    else:
        per_tid_count = defaultdict(int)
        for s in SENTENCES:
            tid = s["template_id"]
            if tid in template_ids and per_tid_count[tid] < limit_per_template:
                out.append(s)
                per_tid_count[tid] += 1
    return out


# ---------------------------------------------------------------------
# Per-sentence driver.
# ---------------------------------------------------------------------

def parse_one(area_config, sentence):
    """Build a fresh brain for the given config and parse ``sentence``.

    Returns {"dependencies": ..., "convergence": ..., "error": None | str}.

    We explicitly drop the brain and force a ``gc.collect()`` afterwards
    because the LEX inner connectome alone is ~400 MB for the n=10000
    configs; without forcing collection, stale brains from prior
    sentences can accumulate in the same worker (Python's generational
    GC is conservative about NumPy arrays) and the worker can be killed
    by the OS for memory pressure, which in turn collapses the whole
    multiprocessing.Pool.
    """
    brain = None
    try:
        brain = ConfigurableEnglishParserBrain(
            p=0.1, area_config=area_config, verbose=False,
        )
        deps, conv = parse_sentence_instrumented(
            brain, sentence, project_rounds=20,
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


def deps_to_set(deps):
    return {tuple(d) for d in deps} if deps else set()


# ---------------------------------------------------------------------
# Multiprocessing scaffolding (top-level so it pickles).
# ---------------------------------------------------------------------

def _worker_init(seed):
    """Initializer called once in each worker process."""
    _CURRENT_SEED[0] = seed


def _worker_parse(args):
    """One sentence on one config. Picklable, top-level."""
    sentence_idx, area_config, sentence_info = args
    r = parse_one(area_config, sentence_info["sentence"])
    return sentence_idx, sentence_info, r


# ---------------------------------------------------------------------
# Per-config aggregation.
# ---------------------------------------------------------------------

def _empty_aggregate():
    return {
        "per_template": defaultdict(lambda: {
            "correct": 0,
            "total": 0,
            "rounds_per_word": [],
        }),
        "per_area_rounds": defaultdict(list),
        "rounds_per_word_all": [],
        "rounds_per_sentence_all": [],
        "failed": [],
        "n_correct": 0,
    }


def _absorb_result(agg, sentence_info, r):
    tid = sentence_info["template_id"]
    sent = sentence_info["sentence"]
    expected_set = {tuple(d) for d in sentence_info["expected_deps"]}

    agg["per_template"][tid]["total"] += 1

    if r["error"] is not None:
        agg["failed"].append({
            "sentence": sent,
            "template_id": tid,
            "expected": [list(d) for d in sorted(expected_set)],
            "got": None,
            "error": r["error"],
        })
        return

    got_deps = r["dependencies"]
    got_set = deps_to_set(got_deps)
    ok = (got_set == expected_set)

    conv = r["convergence"]
    sent_rounds = []
    for call in conv["per_call"]:
        agg["rounds_per_word_all"].append(call["rounds_to_stabilize_max"])
        sent_rounds.append(call["rounds_to_stabilize_max"])
        for area_ext, r_count in call["per_area_rounds"].items():
            agg["per_area_rounds"][area_ext].append(r_count)
    agg["rounds_per_sentence_all"].append(sum(sent_rounds))
    agg["per_template"][tid]["rounds_per_word"].extend(sent_rounds)

    if ok:
        agg["n_correct"] += 1
        agg["per_template"][tid]["correct"] += 1
    else:
        agg["failed"].append({
            "sentence": sent,
            "template_id": tid,
            "expected": [list(d) for d in sorted(expected_set)],
            "got": [list(d) for d in got_deps],
        })


def _finalize_aggregate(config_name, area_config, agg, elapsed):
    total = sum(rec["total"] for rec in agg["per_template"].values())
    n_correct = agg["n_correct"]
    accuracy = (n_correct / total) if total else 0.0

    def _mean_std(xs):
        if not xs:
            return 0.0, 0.0
        if len(xs) == 1:
            return float(xs[0]), 0.0
        return statistics.mean(xs), statistics.stdev(xs)

    mean_rpw, std_rpw = _mean_std(agg["rounds_per_word_all"])
    mean_rps, std_rps = _mean_std(agg["rounds_per_sentence_all"])

    by_template = []
    for tid in sorted(agg["per_template"].keys()):
        rec = agg["per_template"][tid]
        m, _ = _mean_std(rec["rounds_per_word"])
        by_template.append({
            "template_id": tid,
            "description": TEMPLATE_DESCRIPTIONS[tid],
            "complexity": COMPLEXITY.get(tid, "unknown"),
            "correct": rec["correct"],
            "total": rec["total"],
            "accuracy": round(rec["correct"] / rec["total"], 4)
                if rec["total"] else 0.0,
            "mean_rounds_per_word": round(m, 3),
        })

    by_area = {}
    for area_ext in sorted(agg["per_area_rounds"].keys()):
        xs = agg["per_area_rounds"][area_ext]
        m, s = _mean_std(xs)
        by_area[area_ext] = {
            "mean_rounds_to_stabilize": round(m, 3),
            "std_rounds_to_stabilize": round(s, 3),
            "n_observations": len(xs),
        }

    return {
        "config_name": config_name,
        "area_config": area_config,
        "overall": {
            "accuracy": round(accuracy, 4),
            "total_correct": n_correct,
            "total_sentences": total,
            "mean_rounds_per_word": round(mean_rpw, 3),
            "std_rounds_per_word": round(std_rpw, 3),
            "mean_rounds_per_sentence": round(mean_rps, 3),
            "std_rounds_per_sentence": round(std_rps, 3),
        },
        "by_template": by_template,
        "by_area": by_area,
        "failed_sentences": agg["failed"],
        "wall_clock_seconds": round(elapsed, 1),
    }


def aggregate_run(config_name, area_config, sentences, *,
                  workers=1, seed=0):
    """Run all ``sentences`` under one config; return an aggregated dict."""
    agg = _empty_aggregate()
    t0 = time.time()

    if workers <= 1:
        # Serial path
        for i, sinfo in enumerate(sentences):
            r = parse_one(area_config, sinfo["sentence"])
            _absorb_result(agg, sinfo, r)
            if (i + 1) % 20 == 0 or (i + 1) == len(sentences):
                elapsed = time.time() - t0
                print(
                    f"  [{config_name}] {i + 1}/{len(sentences)} "
                    f"correct={agg['n_correct']} elapsed={elapsed:.0f}s",
                    flush=True,
                )
    else:
        # Parallel path. Use 'spawn' so workers re-import this module
        # and re-apply the seed patch cleanly.
        ctx = mp.get_context("spawn")
        n = len(sentences)
        printed_done = 0
        args = [(i, area_config, s) for i, s in enumerate(sentences)]
        with ctx.Pool(
            workers,
            initializer=_worker_init,
            initargs=(seed,),
        ) as pool:
            # imap_unordered for responsiveness; we re-sort by sentence
            # index after if needed (we don't, _absorb_result is order-
            # independent for stats).
            for sentence_idx, sinfo, r in pool.imap_unordered(
                _worker_parse, args, chunksize=1,
            ):
                _absorb_result(agg, sinfo, r)
                printed_done += 1
                if printed_done % 20 == 0 or printed_done == n:
                    elapsed = time.time() - t0
                    print(
                        f"  [{config_name}] {printed_done}/{n} "
                        f"correct={agg['n_correct']} "
                        f"elapsed={elapsed:.0f}s "
                        f"({workers} workers)",
                        flush=True,
                    )

    elapsed = time.time() - t0
    return _finalize_aggregate(config_name, area_config, agg, elapsed)


# ---------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------

CONFIG_DISPLAY_ORDER = [
    "baseline",
    "experiment_A_verb_enlarged",
    "experiment_B_closed_class_shrunk",
    "experiment_C_proportional",
    "experiment_D_inverted",
]


def print_round_summary(round_label, n_sentences, all_results):
    line_w = 70
    print()
    print(f"{round_label} RESULTS ({n_sentences} sentences per config)")
    print("=" * line_w)

    print()
    print(f"{'Config':<28s} {'Accuracy':>9s} {'Rds/word':>10s} {'Rds/word(std)':>14s}")
    print("-" * line_w)
    for name in CONFIG_DISPLAY_ORDER:
        if name not in all_results:
            continue
        r = all_results[name]
        ov = r["overall"]
        print(f"{SHORT_NAMES[name]:<28s} "
              f"{ov['accuracy'] * 100:>8.1f}% "
              f"{ov['mean_rounds_per_word']:>10.2f} "
              f"{ov['std_rounds_per_word']:>14.2f}")

    area_order = ["VERB", "SUBJ", "OBJ", "DET", "ADJ", "ADV", "PREP", "PREPP"]
    print()
    print(f"Per-area convergence rounds (mean) -- {round_label}:")
    header = f"{'Area':<8s}"
    for name in CONFIG_DISPLAY_ORDER:
        if name not in all_results:
            continue
        header += f"{SHORT_NAMES[name]:>22s}"
    print(header)
    print("-" * len(header))
    for area in area_order:
        row = f"{area:<8s}"
        for name in CONFIG_DISPLAY_ORDER:
            if name not in all_results:
                continue
            r = all_results[name]
            v = r["by_area"].get(area, {}).get("mean_rounds_to_stabilize")
            row += f"{(f'{v:.2f}' if v is not None else '-'):>22s}"
        print(row)

    print()
    print(f"Accuracy by template complexity -- {round_label}:")
    header = f"{'Complexity':<12s}"
    for name in CONFIG_DISPLAY_ORDER:
        if name not in all_results:
            continue
        header += f"{SHORT_NAMES[name]:>22s}"
    print(header)
    print("-" * len(header))
    for bucket in ("simple", "medium", "complex"):
        row = f"{bucket:<12s}"
        for name in CONFIG_DISPLAY_ORDER:
            if name not in all_results:
                continue
            r = all_results[name]
            buckets = [t for t in r["by_template"] if t["complexity"] == bucket]
            if not buckets:
                row += f"{'-':>22s}"
                continue
            corr = sum(t["correct"] for t in buckets)
            tot = sum(t["total"] for t in buckets)
            pct = (corr / tot * 100.0) if tot else 0.0
            row += f"{(f'{pct:.1f}%' + f' ({corr}/{tot})'):>22s}"
        print(row)
    print()


# ---------------------------------------------------------------------
# CLI + driver.
# ---------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Heterogeneous Neuron Density experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Run a single config by name. Default: all 5.")
    p.add_argument("--round1-only", action="store_true",
                   help="Run only Round 1 (6 priority templates).")
    p.add_argument("--full-only", action="store_true",
                   help="Run only the full 18-template pass; skip Round 1.")
    p.add_argument("--seed", type=int, default=0,
                   help="Brain RNG seed and PYTHONHASHSEED (default 0).")
    p.add_argument("--limit-per-template", type=int, default=None,
                   help="Cap sentences per template (for fast smoke tests).")
    default_workers = min(os.cpu_count() or 1, 8)
    p.add_argument(
        "--workers", type=int, default=default_workers,
        help=f"Parallel sentence workers (default {default_workers} on this machine). "
             "Set 1 for serial.",
    )
    return p.parse_args()


def _selected_configs(args):
    if args.config:
        if args.config not in CONFIGS:
            print(f"Unknown config: {args.config!r}", file=sys.stderr)
            print(f"Choices: {list(CONFIGS.keys())}", file=sys.stderr)
            sys.exit(2)
        return [args.config]
    return list(CONFIGS.keys())


def _ensure_results_dir():
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    return out


def _save_result(out_dir, config_name, label, result):
    path = os.path.join(out_dir, f"{config_name}_{label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {path}")


def _run_label(label, template_ids, configs_to_run, out_dir, args):
    sentences = _filter_sentences(template_ids,
                                  limit_per_template=args.limit_per_template)
    print(f"\n=== {label.upper()} :: "
          f"{len(sentences)} sentences "
          f"({len(template_ids)} templates) "
          f"x {len(configs_to_run)} configs ===")

    all_results = {}
    for cname in configs_to_run:
        cfg = CONFIGS[cname]
        validate_config(cname, cfg)
        print(f"\n--- Running {cname} ({SHORT_NAMES[cname]}) ---")
        r = aggregate_run(
            cname, cfg, sentences,
            workers=args.workers, seed=args.seed,
        )
        all_results[cname] = r
        _save_result(out_dir, cname, label, r)

    return all_results, len(sentences)


def main():
    args = _parse_args()

    # PYTHONHASHSEED has to be set in env BEFORE Python starts using
    # string hashing. If we were not launched with the right value,
    # re-exec the parent. Workers inherit PYTHONHASHSEED from our env
    # and so do not need to re-exec.
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    _CURRENT_SEED[0] = args.seed

    validate_all()
    configs_to_run = _selected_configs(args)
    out_dir = _ensure_results_dir()

    print(f"Configs to run: {configs_to_run}")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}; "
          f"brain seed: {args.seed}; workers: {args.workers}")
    print(f"Output dir: {out_dir}")
    if args.limit_per_template is not None:
        print(f"limit_per_template = {args.limit_per_template}")

    if args.round1_only and args.full_only:
        print("--round1-only and --full-only are mutually exclusive.",
              file=sys.stderr)
        sys.exit(2)

    if not args.full_only:
        round1_results, n_r1 = _run_label(
            "round1", ROUND1_TEMPLATES, configs_to_run, out_dir, args,
        )
        print_round_summary("ROUND 1", n_r1, round1_results)

        if args.round1_only:
            print("--round1-only: stopping before full run.")
            return

        print("Proceeding to full 18-template run...")

    all_active = [tid for tid in TEMPLATE_DESCRIPTIONS
                  if tid not in (14, 20)]
    full_results, n_full = _run_label(
        "full", all_active, configs_to_run, out_dir, args,
    )
    print_round_summary("FULL", n_full, full_results)


if __name__ == "__main__":
    main()
