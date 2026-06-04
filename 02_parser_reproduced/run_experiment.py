

import argparse
import ast
import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict



def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the Brain (and PYTHONHASHSEED). "
                        "Default 0.")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated list of seeds for a multi-seed "
                        "run. If given, --seed is ignored and the script "
                        "runs the experiment once per seed and reports "
                        "per-seed + mean/std summary.")
    p.add_argument("--out-name", type=str, default=None,
                   help="Override the output JSON filename (without "
                        "directory). Default: "
                        "baseline_parser_reproduction.json for single seed; "
                        "baseline_parser_reproduction_multiseed.json for "
                        "multi-seed.")
    return p.parse_args()


_ARGS = _parse_args()


_HASH_SEED = (
    int(_ARGS.seeds.split(",")[0]) if _ARGS.seeds else _ARGS.seed
)
if os.environ.get("PYTHONHASHSEED") != str(_HASH_SEED):
    os.environ["PYTHONHASHSEED"] = str(_HASH_SEED)
    os.execv(sys.executable, [sys.executable] + sys.argv)




HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import brain as ac_brain  
import parser as ac_parser  
from test_sentences import (  
    SENTENCES, TEMPLATE_DESCRIPTIONS, ACTIVE_TEMPLATES, EXCLUDED_TEMPLATES,
)



_CURRENT_SEED = [_HASH_SEED]


def _install_seed_patch():
    
    _orig = ac_brain.Brain.__init__

    def _seeded_init(self, p, *args, **kwargs):
        kwargs.pop("seed", None)
        _orig(self, p, *args, seed=_CURRENT_SEED[0], **kwargs)

    ac_brain.Brain.__init__ = _seeded_init


_install_seed_patch()



DEPS_RE = re.compile(r"Got dependencies:\s*\n\s*(\[.*\])")


def run_one(sentence: str):
    
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            ac_parser.parse(
                sentence=sentence,
                language="English",
                verbose=False,
                debug=False,
                
            )
    except Exception as e:  
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    out = buf.getvalue()
    m = DEPS_RE.search(out)
    if not m:
        return None, f"No 'Got dependencies' line found. tail:\n{out[-400:]}"
    try:
        return ast.literal_eval(m.group(1)), None
    except Exception as e:  # noqa: BLE001
        return None, f"Failed to parse deps list: {e}; raw={m.group(1)!r}"


def deps_to_set(deps):
    return {tuple(d) for d in deps} if deps else set()




def run_full_pass(seed: int):
    
    _CURRENT_SEED[0] = seed

    per_template = defaultdict(lambda: {"correct": 0, "total": 0})
    per_sentence = []

    t0 = time.time()
    for i, sent_info in enumerate(SENTENCES):
        sent = sent_info["sentence"]
        tid = sent_info["template_id"]
        expected = [tuple(d) for d in sent_info["expected_deps"]]
        expected_set = set(expected)

        per_template[tid]["total"] += 1
        deps, err = run_one(sent)

        if err is not None:
            ok = False
            per_sentence.append({
                "sentence": sent,
                "template_id": tid,
                "correct": False,
                "expected_deps": [list(d) for d in expected],
                "got_deps": None,
                "error": err.splitlines()[0],
            })
        else:
            got_set = deps_to_set(deps)
            ok = got_set == expected_set
            per_sentence.append({
                "sentence": sent,
                "template_id": tid,
                "correct": ok,
                "expected_deps": [list(d) for d in expected],
                "got_deps": [list(d) for d in deps],
            })

        if ok:
            per_template[tid]["correct"] += 1

        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            print(f"  [seed={seed}] {i + 1}/{len(SENTENCES)} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    correct = sum(1 for s in per_sentence if s["correct"])
    total = len(SENTENCES)

    return {
        "seed": seed,
        "python_hash_seed": _HASH_SEED,
        "total_sentences": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "wall_clock_seconds": round(elapsed, 1),
        "per_template": [
            {
                "template_id": tid,
                "description": TEMPLATE_DESCRIPTIONS[tid],
                "correct": per_template[tid]["correct"],
                "total": per_template[tid]["total"],
            }
            for tid in ACTIVE_TEMPLATES
        ],
        "per_sentence": per_sentence,
    }




def print_single_seed_report(result):
    print()
    print(f"Seed: {result['seed']}  (PYTHONHASHSEED={result['python_hash_seed']})")
    print(f"Total sentences: {result['total_sentences']}")
    print(f"Correct parses:  {result['correct']}")
    print(f"Accuracy:        {result['accuracy'] * 100:.1f}%")
    print()
    print("Results by template:")
    for tid in range(1, 21):
        desc = TEMPLATE_DESCRIPTIONS[tid]
        label = f"Template {tid:<2d} ({desc}):"
        if tid in EXCLUDED_TEMPLATES:
            print(f"  {label:<40s} EXCLUDED (requires unimplemented COMP-extension)")
            continue
        rec = next(t for t in result["per_template"] if t["template_id"] == tid)
        print(f"  {label:<40s} {rec['correct']}/{rec['total']}")
    print()
    failures = [s for s in result["per_sentence"] if not s["correct"]]
    if failures:
        print(f"Failed sentences ({len(failures)}):")
        for s in failures:
            if s.get("error"):
                print(f'  - "{s["sentence"]}" -- error: {s["error"]}')
            else:
                exp = {tuple(d) for d in s["expected_deps"]}
                got = {tuple(d) for d in s["got_deps"]}
                print(f'  - "{s["sentence"]}"')
                print(f"      missing: {sorted(exp - got)}")
                print(f"      extra:   {sorted(got - exp)}")
    else:
        print("Failed sentences: none")
    print()
    print(f"Wall-clock: {result['wall_clock_seconds']}s")


def print_multiseed_summary(results):
    import statistics

    print()
    print("=" * 60)
    print("Multi-seed summary")
    print("=" * 60)
    print(f"PYTHONHASHSEED:    {results[0]['python_hash_seed']} (pinned)")
    print(f"Seeds run:         {[r['seed'] for r in results]}")
    n = results[0]["total_sentences"]
    accs = [r["accuracy"] for r in results]
    correct = [r["correct"] for r in results]
    mean_acc = statistics.mean(accs)
    stdev_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
    mean_corr = statistics.mean(correct)
    stdev_corr = statistics.stdev(correct) if len(correct) > 1 else 0.0
    print(f"Sentences / seed:  {n}")
    print(f"Correct (mean):    {mean_corr:.2f} +/- {stdev_corr:.2f}")
    print(f"Accuracy (mean):   {mean_acc * 100:.2f}% +/- {stdev_acc * 100:.2f}%")
    print()
    print("Per-seed:")
    for r in results:
        print(f"  seed={r['seed']:<4d}  {r['correct']}/{n}  "
              f"({r['accuracy'] * 100:.1f}%)  "
              f"{r['wall_clock_seconds']:.0f}s")
    print()
    
    print("Per-template accuracy (mean +/- std across seeds):")
    for tid in range(1, 21):
        desc = TEMPLATE_DESCRIPTIONS[tid]
        label = f"Template {tid:<2d} ({desc}):"
        if tid in EXCLUDED_TEMPLATES:
            print(f"  {label:<40s} EXCLUDED")
            continue
        per_seed_correct = [
            next(t for t in r["per_template"] if t["template_id"] == tid)["correct"]
            for r in results
        ]
        m = statistics.mean(per_seed_correct)
        sd = statistics.stdev(per_seed_correct) if len(per_seed_correct) > 1 else 0.0
        print(f"  {label:<40s} {m:.2f}/10 +/- {sd:.2f}")




def main():
    out_dir = os.path.join(REPO_ROOT, "parser_reproduced", "results_reproduced")
    os.makedirs(out_dir, exist_ok=True)

    if _ARGS.seeds:
        seeds = [int(s) for s in _ARGS.seeds.split(",")]
        all_results = []
        for s in seeds:
            print(f"\n=== Running seed {s} ===", flush=True)
            r = run_full_pass(s)
            print_single_seed_report(r)
            all_results.append(r)

        print_multiseed_summary(all_results)

        out_name = _ARGS.out_name or "baseline_parser_reproduction_multiseed.json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w") as f:
            json.dump({
                "python_hash_seed": _HASH_SEED,
                "seeds": seeds,
                "runs": all_results,
            }, f, indent=2)
        print(f"\nWrote {out_path}")

    else:
        r = run_full_pass(_ARGS.seed)
        print_single_seed_report(r)
        
        r["excluded_templates"] = [
            {
                "template_id": tid,
                "description": TEMPLATE_DESCRIPTIONS[tid],
                "reason": (
                    "Requires unimplemented COMP-style chained-modifier "
                    "extension (Paper Section 6, 'big bad problem'). "
                    "Not addressable by n/k allocation."
                ),
            }
            for tid in EXCLUDED_TEMPLATES
        ]
        out_name = _ARGS.out_name or "baseline_parser_reproduction.json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
