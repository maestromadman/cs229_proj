

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import nemo_extensions as nx


def run(n=nx.N_DEFAULT, k=nx.K_DEFAULT, p=nx.P_DEFAULT, m=3, w_ref=1.5,
        w1_lo=1.3, w1_hi=1.7, n_weights=20, n_graphs=30, n_trials=500,
        internal_factor=nx.INTERNAL_FACTOR, noise_scale=nx.NOISE_SCALE_DEFAULT,
        seed=0, verbose=True):
    
    sigma = nx.noise_std(k, p, noise_scale)
    w1_values = np.linspace(w1_lo, w1_hi, n_weights)
    results = np.zeros((n_graphs, n_weights), dtype=np.float64)
    master = np.random.default_rng(seed)

    if verbose:
        print(f"[1a] m={m} assemblies, w_ref={w_ref}, internal={internal_factor}, "
              f"noise std={sigma:.2f}")
        print(f"     {n_graphs} graphs x {n_weights} weights x {n_trials} trials, "
              f"n={n}, k={k}")
    t0 = time.time()
    for g in range(n_graphs):
        graph = nx.make_meanfield_graph(n, k, p, m, rng=master)
        rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        for wi, w1 in enumerate(w1_values):
            weights = [w1] + [w_ref] * (m - 1)
            results[g, wi] = nx.win_probability(
                graph, weights, sigma, n_trials, rng, target=0,
                internal_factor=internal_factor)
        if verbose and (g + 1) % max(1, n_graphs // 5) == 0:
            print(f"     graph {g+1}/{n_graphs}  ({time.time()-t0:.1f}s)")
    return w1_values, results


def plot(w1_values, results, w_ref, out_path, title_suffix=""):
    mean = results.mean(axis=0)
    q05 = np.quantile(results, 0.05, axis=0)
    q95 = np.quantile(results, 0.95, axis=0)

    
    weights_matrix = np.column_stack([w1_values,
                                      np.full_like(w1_values, w_ref),
                                      np.full_like(w1_values, w_ref)])
    best_lam, fit = nx.fit_softmax_lambda(weights_matrix, mean)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.fill_between(w1_values, q05, q95, color="#9ecae1", alpha=0.6,
                    label="[0.05, 0.95] across graphs")
    ax.plot(w1_values, mean, color="#08519c", lw=2, marker="o", ms=4,
            label="Estimated")
    ax.plot(w1_values, fit, color="k", ls="--", lw=1.6,
            label=fr"Target (softmax, $\lambda$={best_lam:.1f})")
    ax.set_xlabel("Weight of Assembly A")
    ax.set_ylabel("Probability of Win")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Fig. 3(a): weight sweep" + title_suffix)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[1a] best-fit softmax lambda = {best_lam:.2f}")
    print(f"[1a] wrote {out_path}")
    return best_lam


def main():
    ap = argparse.ArgumentParser(description="Experiment 1a (Fig 3a)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graphs", type=int, default=30)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--weights", type=int, default=20)
    ap.add_argument("--w-ref", type=float, default=1.5,
                    help="input weight of A2,A3 (internal weight stays 2)")
    ap.add_argument("--full", action="store_true",
                    help="paper-scale: 100 graphs, 1000 trials")
    ap.add_argument("--out", default="figure_3a.png")
    args = ap.parse_args()

    n_graphs, n_trials = args.graphs, args.trials
    if args.full:
        n_graphs, n_trials = 100, 1000

    w1_values, results = run(w_ref=args.w_ref, n_weights=args.weights,
                             n_graphs=n_graphs, n_trials=n_trials, seed=args.seed)
    plot(w1_values, results, args.w_ref, args.out)


if __name__ == "__main__":
    main()
