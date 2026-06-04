

from __future__ import annotations

import argparse
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_1a
import experiment_1b
import experiment_1c
import experiment_2
import experiment_3


def main():
    ap = argparse.ArgumentParser(description="Run all coin-flip experiments")
    ap.add_argument("--full", action="store_true",
                    help="paper-scale repetition counts (100 graphs, 1000 trials)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.full:
        graphs_1, trials_1 = 100, 1000
        graphs_1bc, trials_1bc = 100, 500
        graphs_5, trials_5 = 20, 500
    else:
        graphs_1, trials_1 = 30, 500
        graphs_1bc, trials_1bc = 30, 500
        graphs_5, trials_5 = 20, 500

    t0 = time.time()

    print("=" * 70 + "\nFigure 3(a): weight sweep\n" + "=" * 70)
    w1, res = experiment_1a.run(n_graphs=graphs_1, n_trials=trials_1,
                                seed=args.seed)
    experiment_1a.plot(w1, res, 1.5, "figure_3a.png")

    print("\n" + "=" * 70 + "\nFigure 3(b): plasticity learns frequency\n" + "=" * 70)
    T1, res, tgt = experiment_1b.run(n_graphs=graphs_1bc, n_trials=trials_1bc,
                                     seed=args.seed)
    experiment_1b.plot(T1, res, tgt, "figure_3b.png")

    print("\n" + "=" * 70 + "\nFigure 4(a,b): varying number of assemblies\n" + "=" * 70)
    mv, ra, ta = experiment_1c.run(experiment_1c._T1_schedule_a,
                                   n_graphs=graphs_1bc, n_trials=trials_1bc,
                                   seed=args.seed, label="a")
    _, rb, tb = experiment_1c.run(experiment_1c._T1_schedule_b,
                                  n_graphs=graphs_1bc, n_trials=trials_1bc,
                                  seed=args.seed + 1, label="b")
    experiment_1c.plot_pair(mv, ra, ta, rb, tb, "figure_4ab.png")

    print("\n" + "=" * 70 + "\nFigure 5: error vs cap size\n" + "=" * 70)
    devs = experiment_2.run(n_graphs=graphs_5, n_trials=trials_5,
                            engine="meanfield", seed=args.seed)
    experiment_2.plot(devs, trials_5, graphs_5, "figure_5.png", engine="meanfield")

    print("\n" + "=" * 70 + "\nFigure 8(b): trigram language model\n" + "=" * 70)
    
    tokens, vocab, trigs, gen, text = experiment_3.run(seed=4)
    with open("figure_8b.txt", "w") as f:
        f.write(text.replace(" . ", ".\n") + "\n\n")
        f.write(f"corpus: {len(tokens)} tokens, {len(trigs)} unique trigrams\n")
        f.write("\nUnique trigrams learned:\n")
        for tg in trigs:
            f.write("  " + " ".join(tg) + "\n")

    print(f"\nAll figures written. Total time {time.time()-t0:.1f}s")
    for fn in ("figure_3a.png", "figure_3b.png", "figure_4ab.png",
               "figure_5.png", "figure_8b.txt"):
        print("  ", fn)


if __name__ == "__main__":
    main()
