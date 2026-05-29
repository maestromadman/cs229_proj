"""experiment_2_fixed_n.py -- Figure 5 with the paper's LITERAL setup (n=25000).

Companion to experiment_2.py.  Same simulator, same dynamics, same plasticity
rule -- the only difference is that n is held at the paper's stated 25,000
across the cap-size sweep instead of being scaled with k.

Paper text (Section 4.1, applied to Section 4.2):
    n = 25000, k swept, p = 0.1, noise std = 5 * sqrt(k*p),
    20 connectivity graphs per k, 500 samples per graph,
    weights to A and B equal (= 2 in the paper, = trained value here).

What this run will show: at fixed n=25000, the curve goes the OPPOSITE
direction from Fig 5 -- error INCREASES with k.  At small k the two
assemblies are tiny (e.g. 2 * 50 neurons of 25,000) and lost among the
background noise tail; they never register in the cap, so neither A nor B
reliably wins and the coin is near-fair (low error).  At large k they
finally register and the residual random-graph asymmetry biases each trial
-> higher error.

This file is preserved as the LITERAL-paper-params reference.  See
`experiment_2.py --n-ratio 10` (constant 20% assembly density) for the
companion run that reproduces Fig 5's *decreasing* shape -- that one
deviates from the literal n but matches the figure.  Document this protocol
gap in the writeup.

Run:
    python3 experiment_2_fixed_n.py --engine torch
    # -> writes figure_5_fixed_n.png
"""

from __future__ import annotations

import argparse

import experiment_2 as e2


def main():
    ap = argparse.ArgumentParser(
        description="Experiment 2 with n=25000 fixed (paper-literal)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--engine", choices=["meanfield", "sparse", "torch"],
                    default="torch")
    ap.add_argument("--device", default="auto",
                    help="torch device: auto|cuda|cpu|mps")
    ap.add_argument("--batch", type=int, default=None,
                    help="torch noise-sample batch size (default: all at once)")
    ap.add_argument("--rounds", type=int, default=20,
                    help="recurrent k-cap rounds (sparse/torch)")
    ap.add_argument("--noise-every-round", action="store_true",
                    help="apply Gaussian noise on every recurrent round, not "
                         "just round 0")
    ap.add_argument("--w", type=float, default=2.0,
                    help="weight on I->A and I->B; paper Fig 5 sets directly to 2")
    ap.add_argument("--ks", type=str, default=None,
                    help="comma-separated cap sizes; defaults to "
                         "experiment_2.K_SWEEP")
    ap.add_argument("--out", default="figure_5_fixed_n.png")
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")] if args.ks else e2.K_SWEEP

    print(f"[fig5-literal] paper-literal params: n=25000 fixed, "
          f"k sweep = {ks}")
    print("[fig5-literal] expect INCREASING error with k (see docstring); "
          "the decreasing-shape reproduction is experiment_2.py --n-ratio 10.")

    devs = e2.run(ks=ks, n=25000, n_graphs=args.graphs, n_trials=args.trials,
                  engine=args.engine, seed=args.seed, device=args.device,
                  batch=args.batch, max_rounds=args.rounds,
                  noise_every_round=args.noise_every_round, w_input=args.w)
    e2.plot(devs, args.trials, args.graphs, args.out, engine=args.engine)


if __name__ == "__main__":
    main()
