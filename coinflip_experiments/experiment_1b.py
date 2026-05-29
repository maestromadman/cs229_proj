"""Experiment 1b -- Assembly coin-flip: plasticity learns frequency (Figure 3b).

Goal: show that the additive Hebbian rule drives the input weights so that, at
test time, Pr(A_i wins) ~ T_i / (T1 + T2 + T3), where T_i is the number of times
assembly A_i was presented during training.

Setup (Dabagia et al. 2024, Fig. 3b):
  * Three outcome assemblies A1, A2, A3, context assembly I; internal weight 2.
  * T2 = T3 = 5 fixed; T1 swept from 1 to 40.
  * Training: present A_i a total of T_i times.  Each presentation fires I then
    A_i and applies the additive rule pi(w)=min(alpha, exp(lambda(1+beta-w))) to
    the I->A_i synapses.  Because every I->A_i synapse starts at the baseline
    weight 1 and receives the same increment, the assembly's input weight
    follows the scalar recurrence w <- w + pi(w) (see Lemma 3), which grows like
    lambda^{-1} log(c T_i) -- exactly the log-growth that turns the test-time
    softmax read-out into proportional sampling.
  * Test: fire I once with noise, run S to convergence (plasticity OFF), repeat
    to estimate Pr(A1 wins).
  * Repeat over random graphs; plot mean + [0.05, 0.95] band against the target
    frequency T1/(T1+T2+T3).

alpha=0.63, beta=0.5, lambda=26 (Sec. 4.1).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import nemo_extensions as nx


def run(n=nx.N_DEFAULT, k=nx.K_DEFAULT, p=nx.P_DEFAULT, m=3, T2=5, T3=5,
        T1_max=40, n_points=20, n_graphs=30, n_trials=500,
        internal_factor=nx.INTERNAL_FACTOR, noise_scale=nx.NOISE_SCALE_DEFAULT,
        alpha=nx.CF_ALPHA, beta=nx.CF_BETA, lam=nx.CF_LAMBDA, seed=0,
        verbose=True):
    sigma = nx.noise_std(k, p, noise_scale)
    T1_values = np.unique(np.round(np.linspace(1, T1_max, n_points))).astype(int)
    n_points = len(T1_values)

    # Trained input weights (graph-independent: scalar Hebbian recurrence).
    w2 = nx.train_assembly_weight(T2, alpha=alpha, beta=beta, lam=lam)
    w3 = nx.train_assembly_weight(T3, alpha=alpha, beta=beta, lam=lam)
    w1_by_T1 = {int(T1): nx.train_assembly_weight(int(T1), alpha=alpha,
                                                  beta=beta, lam=lam)
                for T1 in T1_values}

    results = np.zeros((n_graphs, n_points), dtype=np.float64)
    master = np.random.default_rng(seed)
    if verbose:
        print(f"[1b] T2=T3={T2}, sweep T1 in {list(T1_values)}")
        print(f"     trained w: w(5)={w2:.4f}, w(1)={w1_by_T1[T1_values[0]]:.4f}, "
              f"w({T1_values[-1]})={w1_by_T1[T1_values[-1]]:.4f}")
        print(f"     {n_graphs} graphs x {n_points} T1 x {n_trials} trials")
    t0 = time.time()
    for g in range(n_graphs):
        graph = nx.make_meanfield_graph(n, k, p, m, rng=master)
        rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        for ti, T1 in enumerate(T1_values):
            weights = [w1_by_T1[int(T1)], w2, w3]
            results[g, ti] = nx.win_probability(
                graph, weights, sigma, n_trials, rng, target=0,
                internal_factor=internal_factor)
        if verbose and (g + 1) % max(1, n_graphs // 5) == 0:
            print(f"     graph {g+1}/{n_graphs}  ({time.time()-t0:.1f}s)")
    target = T1_values / (T1_values + T2 + T3)
    return T1_values, results, target


def plot(T1_values, results, target, out_path, title_suffix=""):
    mean = results.mean(axis=0)
    q05 = np.quantile(results, 0.05, axis=0)
    q95 = np.quantile(results, 0.95, axis=0)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.fill_between(T1_values, q05, q95, color="#9ecae1", alpha=0.6,
                    label="[0.05, 0.95] across graphs")
    ax.plot(T1_values, mean, color="#08519c", lw=2, marker="o", ms=4,
            label="Estimated")
    ax.plot(T1_values, target, color="k", ls="--", lw=1.6,
            label=r"$\frac{T_1}{T_1+T_2+T_3}$")
    ax.set_xlabel("Number of Presentations of Assembly A")
    ax.set_ylabel("Probability of Win")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Fig. 3(b): plasticity learns frequency" + title_suffix)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[1b] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Experiment 1b (Fig 3b)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graphs", type=int, default=30)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--full", action="store_true",
                    help="paper-scale: 100 graphs, 500 trials")
    ap.add_argument("--out", default="figure_3b.png")
    args = ap.parse_args()

    n_graphs, n_trials = args.graphs, args.trials
    if args.full:
        n_graphs, n_trials = 100, 500

    T1_values, results, target = run(n_points=args.points, n_graphs=n_graphs,
                                     n_trials=n_trials, seed=args.seed)
    plot(T1_values, results, target, args.out)


if __name__ == "__main__":
    main()
