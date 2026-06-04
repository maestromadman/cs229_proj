
from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import nemo_extensions as nx


def _T1_schedule_a(m):
    return 5 * (m - 1)


def _T1_schedule_b(m):
    return 10


def run(schedule, n=nx.N_DEFAULT, k=nx.K_DEFAULT, p=nx.P_DEFAULT,
        m_values=range(2, 11), T_rest=5, n_graphs=30, n_trials=500,
        internal_factor=nx.INTERNAL_FACTOR, noise_scale=nx.NOISE_SCALE_DEFAULT,
        alpha=nx.CF_ALPHA, beta=nx.CF_BETA, lam=nx.CF_LAMBDA, seed=0, label="",
        verbose=True):
    sigma = nx.noise_std(k, p, noise_scale)
    m_values = list(m_values)
    results = np.zeros((n_graphs, len(m_values)), dtype=np.float64)
    target = np.zeros(len(m_values), dtype=np.float64)
    master = np.random.default_rng(seed)
    w_rest = nx.train_assembly_weight(T_rest, alpha=alpha, beta=beta, lam=lam)

    if verbose:
        print(f"[1c-{label}] m in {m_values}, T_rest={T_rest}, "
              f"{n_graphs} graphs x {n_trials} trials")
    t0 = time.time()
    for mi, m in enumerate(m_values):
        T1 = schedule(m)
        w1 = nx.train_assembly_weight(T1, alpha=alpha, beta=beta, lam=lam)
        weights = [w1] + [w_rest] * (m - 1)
        target[mi] = T1 / (T1 + T_rest * (m - 1))
        for g in range(n_graphs):
            graph = nx.make_meanfield_graph(n, k, p, m, rng=master)
            rng = np.random.default_rng(master.integers(0, 2**63 - 1))
            results[g, mi] = nx.win_probability(
                graph, weights, sigma, n_trials, rng, target=0,
                internal_factor=internal_factor)
        if verbose:
            print(f"   m={m}: T1={T1}, target={target[mi]:.3f}, "
                  f"est={results[:, mi].mean():.3f}  ({time.time()-t0:.1f}s)")
    return np.array(m_values), results, target


def plot_pair(m_vals, res_a, tgt_a, res_b, tgt_b, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, res, tgt, sub in ((axes[0], res_a, tgt_a, "(a)"),
                              (axes[1], res_b, tgt_b, "(b)")):
        mean = res.mean(axis=0)
        q05 = np.quantile(res, 0.05, axis=0)
        q95 = np.quantile(res, 0.95, axis=0)
        ax.fill_between(m_vals, q05, q95, color="#9ecae1", alpha=0.6)
        ax.plot(m_vals, mean, color="#08519c", lw=2, marker="o", ms=4,
                ls=":", label="Estimated")
        ax.plot(m_vals, tgt, color="k", ls="--", lw=1.6, label="Target")
        ax.set_xlabel("Number of assemblies")
        ax.set_ylim(0.0, 0.85)
        ax.set_title(f"Fig. 4{sub}")
        ax.legend(loc="upper right", fontsize=9, frameon=False)
    axes[0].set_ylabel("Probability of win")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[1c] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Experiment 1c (Fig 4a/4b)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graphs", type=int, default=30)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--full", action="store_true",
                    help="paper-scale: 100 graphs, 500 trials")
    ap.add_argument("--out", default="figure_4ab.png")
    args = ap.parse_args()

    n_graphs, n_trials = args.graphs, args.trials
    if args.full:
        n_graphs, n_trials = 100, 500

    m_vals, res_a, tgt_a = run(_T1_schedule_a, n_graphs=n_graphs,
                               n_trials=n_trials, seed=args.seed, label="a")
    _, res_b, tgt_b = run(_T1_schedule_b, n_graphs=n_graphs,
                          n_trials=n_trials, seed=args.seed + 1, label="b")
    plot_pair(m_vals, res_a, tgt_a, res_b, tgt_b, args.out)


if __name__ == "__main__":
    main()
