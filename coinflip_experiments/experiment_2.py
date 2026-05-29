"""Experiment 2 -- Scaling: error vs cap size (Figure 5).

Goal: with two assemblies A, B trained equally (target distribution uniform,
Pr(A wins)=1/2), show that the error |Pr(A wins) - 1/2| converges toward the
irreducible sampling-noise floor as the cap size k grows.

Setup (Dabagia et al. 2024, Fig. 5):
  * Two assemblies A, B, equal input weights (equal training).
  * Sweep k over [50,100,200,500,1000,1500,2000,2500,3000,3500,4000].
  * For each k: 20 random graphs; estimate Pr(A wins) from 500 noise samples per
    graph; record |Pr(A wins) - 0.5|.
  * Plot mean over the 20 trials (dark line), range across trials (shaded), and
    the "Unbiased Maximum" reference = E[max of 20 |Binom(500,1/2)/500-1/2|].

Two engines (`--engine`):
  * meanfield (default, fast): the round-0 noisy-k-cap argmax used by 1a/1b/1c.
    IMPORTANT CAVEAT.  Fig. 5's steep error at *small* k is a genuinely
    NONLINEAR winner-take-all effect: at small k the activation noise
    sigma=5 sqrt(kp) is small, so the recurrent attractor locks deterministically
    onto whichever assembly the fixed graph favors (error ~ 0.5); as k grows,
    sigma grows and washes the asymmetry out (error -> sampling floor).  In any
    *linear* Gaussian read-out the k-dependence of signal and noise cancels
    (we verified this analytically and empirically), so the mean-field engine
    cannot reproduce the steep small-k branch -- it instead reports error that
    stays within a small band near the sampling floor.  It still demonstrates
    the paper's headline conclusion (the learned two-assembly distribution is
    ~uniform, error near the floor) for k >~ 200.
  * sparse (faithful, CPU, slow): builds the real n x n recurrent connectome
    (scipy.sparse) and runs true noisy k-cap to convergence, so the nonlinear
    winner-take-all -- and the full decreasing curve -- emerge.  CPU-only; a
    large job at n=25000 (hours).
  * torch (faithful, GPU): same dynamics as `sparse`, but the connectome is a
    DENSE tensor and noise samples are batched, so each recurrent round is one
    matmul.  At n=25000 the float32 matrix is ~2.5 GB (fits a 24 GB NVIDIA L4)
    and the whole sweep runs in minutes.  Auto-uses CUDA if present.

Running the faithful curve on a cloud GPU (e.g. an NVIDIA L4, 24 GB):
    pip install torch          # the wheel bundles its CUDA runtime; you only
                               # need the NVIDIA driver the instance already has
    # sanity-check the curve direction first (fast):
    python3 experiment_2.py --engine torch --validate
    # then the full sweep:
    python3 experiment_2.py --engine torch --out figure_5_torch.png
Memory note: each graph holds one ~2.5 GB dense matrix; it is freed between
graphs.  Lower --batch if you ever hit an OOM at very large n.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import nemo_extensions as nx


K_SWEEP = [50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]


def _dev_meanfield(k, n, p, w, n_graphs, n_trials, internal_factor,
                   noise_scale, master):
    sigma = nx.noise_std(k, p, noise_scale)
    d = np.zeros(n_graphs)
    for g in range(n_graphs):
        graph = nx.make_meanfield_graph(n, k, p, m=2, rng=master)
        rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        pA = nx.win_probability(graph, [w, w], sigma, n_trials, rng, target=0,
                                internal_factor=internal_factor)
        d[g] = abs(pA - 0.5)
    return d


def _dev_sparse(k, n, p, w, n_graphs, n_trials, internal_factor,
                noise_scale, master):
    sigma = nx.noise_std(k, p, noise_scale)
    d = np.zeros(n_graphs)
    for g in range(n_graphs):
        area = nx.build_sparse_area(n, k, p, m=2, rng=master,
                                    internal_factor=internal_factor)
        rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        wins = sum(nx.sample_winner_sparse(area, [w, w], sigma, rng) == 0
                   for _ in range(n_trials))
        d[g] = abs(wins / n_trials - 0.5)
    return d


def _dev_torch(k, n, p, w, n_graphs, n_trials, internal_factor, noise_scale,
               master, device, batch, max_rounds, noise_every_round):
    import torch
    sigma = nx.noise_std(k, p, noise_scale)
    d = np.zeros(n_graphs)
    for g in range(n_graphs):
        seed = int(master.integers(0, 2**31 - 1))
        tgen = torch.Generator(device=device); tgen.manual_seed(seed)
        np_rng = np.random.default_rng(seed)
        W, in_deg, asm = nx.build_dense_area_torch(
            n, k, p, 2, device, internal_factor=internal_factor,
            np_rng=np_rng, torch_gen=tgen)
        winners = nx.sample_winners_torch(
            W, in_deg, asm, [w, w], k, sigma, n_trials, max_rounds=max_rounds,
            batch=batch, torch_gen=tgen,
            noise_every_round=noise_every_round)
        pA = (winners == 0).to(torch.float32).mean().item()
        d[g] = abs(pA - 0.5)
        del W, in_deg, asm, winners
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return d


def run(ks=K_SWEEP, n=nx.N_DEFAULT, p=nx.P_DEFAULT, n_graphs=20, n_trials=500,
        train_presentations=5, internal_factor=nx.INTERNAL_FACTOR,
        noise_scale=nx.NOISE_SCALE_DEFAULT, engine="meanfield", seed=0,
        device="auto", batch=None, max_rounds=20, n_ratio=None,
        noise_every_round=False, w_input=None, verbose=True):
    # n_ratio: if set, neurons scale with cap as n_k = round(n_ratio * k), holding
    # assembly density constant (the paper's n/k = 25000/500 = 50).  This is the
    # regime in which Fig 5's decreasing curve appears; fixed n gives the opposite.
    # w_input: weight on I -> A and I -> B; if None, use the trained value (Fig 3b
    # convention); the paper Fig 5 sets it directly to 2 (the "here, 2" in §4.2).
    w = (float(w_input) if w_input is not None
         else nx.train_assembly_weight(train_presentations))   # equal weights A,B
    master = np.random.default_rng(seed)
    if engine == "torch":
        dev = nx.torch_device(device)
        def dev_fn(k, n, p, w, ng, nt, ifac, nscale, m):
            return _dev_torch(k, n, p, w, ng, nt, ifac, nscale, m,
                              dev, batch, max_rounds, noise_every_round)
        engine_desc = f"torch[{dev.type}]" + (" +noise/round" if noise_every_round else "")
    elif engine == "sparse":
        dev_fn = _dev_sparse
        engine_desc = "sparse[cpu]"
    else:
        dev_fn = _dev_meanfield
        engine_desc = "meanfield"
    devs = {}
    if verbose:
        n_desc = f"n={n_ratio:g}*k" if n_ratio else f"n={n}"
        print(f"[2] engine={engine_desc}, equal weights w={w:.4f}, {n_desc}, "
              f"{n_graphs} graphs/k, {n_trials} trials/graph")
    t0 = time.time()
    for k in ks:
        n_k = int(round(n_ratio * k)) if n_ratio else n
        if engine == "torch" and n_k * n_k * 4 > 20e9:
            print(f"   k={k}: SKIP (dense {n_k}x{n_k} float32 = "
                  f"{n_k*n_k*4/1e9:.0f} GB > ~20 GB GPU budget; "
                  f"use --engine sparse or a smaller --n-ratio)")
            continue
        devs[k] = dev_fn(k, n_k, p, w, n_graphs, n_trials, internal_factor,
                         noise_scale, master)
        if verbose:
            d = devs[k]
            ntag = f" n={n_k}" if n_ratio else ""
            print(f"   k={k:<5d}{ntag} mean|err|={d.mean():.4f} "
                  f"min={d.min():.4f} max={d.max():.4f} ({time.time()-t0:.1f}s)")
    return devs


def plot(devs, n_trials, n_graphs, out_path, engine="meanfield", n_ratio=None):
    ks = sorted(devs.keys())
    mean = np.array([devs[k].mean() for k in ks])
    lo = np.array([devs[k].min() for k in ks])
    hi = np.array([devs[k].max() for k in ks])
    unbiased = nx.unbiased_max_deviation(n_trials, n_graphs)

    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    ax.fill_between(ks, lo, hi, color="#9ecae1", alpha=0.6, label="range across trials")
    ax.plot(ks, mean, color="#08519c", lw=2, marker="o", ms=4,
            label="Empirical Mean Error")
    ax.axhline(unbiased, color="k", ls="--", lw=1.5,
               label=f"Unbiased Maximum ({unbiased:.3f})")
    ax.set_xlabel("Cap Size")
    ax.set_ylabel("Error in Proportion")
    ax.set_ylim(0.0, max(hi.max(), 0.5) * 1.05)
    dens = f", n={n_ratio:g}k (const density)" if n_ratio else f", fixed n"
    ax.set_title(f"Fig. 5: error vs cap size  [{engine}{dens}]")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    if engine == "meanfield":
        ax.text(0.5, 0.02,
                "mean-field: error band near sampling floor; steep small-k\n"
                "branch needs --engine sparse/torch (nonlinear winner-take-all)",
                transform=ax.transAxes, fontsize=7, color="0.4", ha="center")
    elif not n_ratio:
        ax.text(0.5, 0.02,
                "fixed n: error RISES with k (small-k assemblies don't register\n"
                "in the cap). For the paper's decreasing shape add --n-ratio 10.",
                transform=ax.transAxes, fontsize=7, color="0.4", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[2] unbiased-maximum reference = {unbiased:.4f}")
    print(f"[2] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Experiment 2 (Fig 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graphs", type=int, default=20)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--engine", choices=["meanfield", "sparse", "torch"],
                    default="meanfield")
    ap.add_argument("--device", default="auto",
                    help="torch engine device: auto|cuda|cpu|mps")
    ap.add_argument("--batch", type=int, default=None,
                    help="torch noise-sample batch size (default: all at once)")
    ap.add_argument("--rounds", type=int, default=20,
                    help="recurrent k-cap rounds (sparse/torch engines)")
    ap.add_argument("--noise-every-round", action="store_true",
                    help="apply Gaussian noise on every recurrent round (the "
                         "general update x(t+1)=k-cap(Wx+z(t))), not just round 0; "
                         "should lower the large-k plateau toward the sampling floor")
    ap.add_argument("--w", type=float, default=2.0,
                    help="weight on I->A and I->B; paper Fig 5 sets it directly "
                         "to 2 (default). Pass a smaller value to use a trained "
                         "weight instead.")
    ap.add_argument("--n", type=int, default=nx.N_DEFAULT,
                    help="neurons per area (used when --n-ratio is not set)")
    ap.add_argument("--n-ratio", type=float, default=None,
                    help="scale neurons with cap: n = n_ratio*k (paper n/k=50). "
                         "Holds assembly density fixed -- the regime where Fig 5's "
                         "decreasing curve appears.")
    ap.add_argument("--ks", type=str, default=None,
                    help="comma-separated cap sizes, e.g. 50,100,200,500,1000 "
                         "(overrides the default sweep)")
    ap.add_argument("--validate", action="store_true",
                    help="quick curve-direction check: 3 cap sizes, few "
                         "graphs/trials, no plot")
    ap.add_argument("--out", default="figure_5.png")
    args = ap.parse_args()
    ks_arg = [int(x) for x in args.ks.split(",")] if args.ks else None

    if args.validate:
        # With a fixed ratio, large k blows up the dense matrix; pick cap sizes
        # whose n stays within a ~24 GB GPU.
        ks = ks_arg or ([50, 200, 1000] if args.n_ratio else [50, 500, 4000])
        ng = args.graphs if args.graphs != 20 else 4
        nt = args.trials if args.trials != 500 else 200
        ntag = f"n={args.n_ratio:g}*k" if args.n_ratio else f"n={args.n}"
        print(f"[2] VALIDATE: engine={args.engine}, {ntag}, k in {ks}, "
              f"{ng} graphs, {nt} trials")
        devs = run(ks=ks, n=args.n, n_graphs=ng, n_trials=nt,
                   engine=args.engine, seed=args.seed, device=args.device,
                   batch=args.batch, max_rounds=args.rounds, n_ratio=args.n_ratio,
                   noise_every_round=args.noise_every_round, w_input=args.w)
        got = sorted(devs)
        means = [devs[k].mean() for k in got]
        trend = ("DECREASING (matches Fig 5)"
                 if means[0] > means[-1] + 0.03
                 else "flat/increasing (does NOT match Fig 5)")
        cells = ", ".join(f"k={k} -> {m:.3f}" for k, m in zip(got, means))
        print(f"[2] mean|err|: {cells}  => {trend}")
        return

    devs = run(ks=ks_arg or K_SWEEP, n=args.n, n_graphs=args.graphs,
               n_trials=args.trials, engine=args.engine, seed=args.seed,
               device=args.device, batch=args.batch, max_rounds=args.rounds,
               n_ratio=args.n_ratio, noise_every_round=args.noise_every_round,
               w_input=args.w)
    plot(devs, args.trials, args.graphs, args.out, engine=args.engine,
         n_ratio=args.n_ratio)


if __name__ == "__main__":
    main()
