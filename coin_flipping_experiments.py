"""Coin-Flipping in the Brain (Dabagia et al. 2024) — experiments.

Implements two experiments built on top of the modifications to brain.py
(Gaussian noise on activations + coin-flipping plasticity rule):

  Step 3: Baseline reproduction of Figure 5 — two competing assemblies A, B
          driven by a context assembly I in a single area, sweeping the
          uniform cap size k.
  Step 4: Novel heterogeneous-k experiment — A, B in a token area of size
          k_tok competing through a category area with k_cat = 500.

Run:  python3 coin_flipping_experiments.py [--pilot]
"""

import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


# -- Coin-Flipping plasticity parameters (Dabagia et al. 2024) ---------------
CF_ALPHA = 0.63
CF_BETA = 0.5
CF_LAMBDA = 26.0
W0 = 2.0   # pre-strengthened weight for assembly edges
P_CONN = 0.1
NOISE_SCALE = 5.0


def coin_flip_pi(w):
    """pi(w) = min(alpha, exp(lambda * (1 + beta - w))) — vectorized."""
    # exp(lambda * 1.5) overflows float32; clamp argument so np.minimum
    # picks alpha for any w well below the saturation threshold.
    arg = np.clip(CF_LAMBDA * (1.0 + CF_BETA - w), a_min=None, a_max=50.0)
    return np.minimum(CF_ALPHA, np.exp(arg))


def apply_cf_update(W):
    """In-place coin-flipping update on existing edges only (W > 0)."""
    mask = W > 0
    if mask.any():
        W[mask] = W[mask] + coin_flip_pi(W[mask])


# ---------------------------------------------------------------------------
# Step 3: Baseline experiment (uniform cap-size sweep)
# ---------------------------------------------------------------------------

def run_baseline_graph(k, T, n_trials, p=P_CONN, rng=None):
    """One graph realization at cap size k.

    Memory note: rather than allocating a full n x n connectome, we maintain
    only I->A and I->B weight matrices.  Because A, B are the only assemblies
    pre-strengthened to w=2 and the "rest of M" has no pre-strengthened edges
    of its own, the A-vs-B competition is determined entirely by the relative
    inputs from I to A and to B (modulo equal additive background noise that
    cancels between groups).  We therefore restrict the top-k k-WTA to
    A union B (size 2k).  The deviation from 0.5 measured by this simulation
    is faithful to the random-graph asymmetry that drives Figure 5.

    Returns the absolute deviation |P(A wins) - 0.5|.
    """
    if rng is None:
        rng = np.random.default_rng()

    # I -> A edges (Bernoulli p) with initial weight W0 on existing edges.
    W_IA = (rng.random((k, k)) < p).astype(np.float32) * W0
    W_IB = (rng.random((k, k)) < p).astype(np.float32) * W0

    # Training: alternate firing I->A and I->B (balanced presentations).
    for _ in range(T):
        apply_cf_update(W_IA)
        apply_cf_update(W_IB)

    # Measurement: I (k neurons, all firing) drives A and B. Per-target
    # input is sum over the rows of the I-to-X weight matrix.
    input_to_A = W_IA.sum(axis=0)  # shape (k,)
    input_to_B = W_IB.sum(axis=0)  # shape (k,)

    noise_std = NOISE_SCALE * math.sqrt(k * p)

    # Vectorize 500 trials.  Noise has shape (n_trials, 2k).
    noise = rng.normal(0.0, noise_std, size=(n_trials, 2 * k)).astype(np.float32)
    combined_signal = np.concatenate([input_to_A, input_to_B])
    totals = combined_signal[None, :] + noise   # (n_trials, 2k)

    # Top-k of 2k for each trial.  Count how many are from A (idx < k).
    top_idx = np.argpartition(totals, 2 * k - k, axis=1)[:, -k:]
    n_A_per_trial = (top_idx < k).sum(axis=1)

    # "A wins" if more of the top-k are from A than from B (tie -> not A).
    a_wins = int((n_A_per_trial > k - n_A_per_trial).sum())
    p_A = a_wins / float(n_trials)
    return abs(p_A - 0.5)


def baseline_sweep(ks, n_graphs, T, n_trials, master_seed=0):
    """Return dict mapping k -> 1-D numpy array of deviations (n_graphs,)."""
    print(f"[baseline] plasticity_rule=coin_flipping, with Gaussian noise "
          f"(std=5*sqrt(k*p)), T={T} train rounds, {n_trials} trials/graph, "
          f"{n_graphs} graphs/k")
    results = {}
    rng_master = np.random.default_rng(master_seed)
    for k in ks:
        t0 = time.time()
        devs = np.zeros(n_graphs, dtype=np.float32)
        for g in range(n_graphs):
            rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
            devs[g] = run_baseline_graph(k, T, n_trials, rng=rng)
        dt = time.time() - t0
        print(f"  k={k:<5d}  mean|err|={devs.mean():.4f}  "
              f"min={devs.min():.4f}  max={devs.max():.4f}  "
              f"({dt:.1f}s)")
        results[k] = devs
    return results


# ---------------------------------------------------------------------------
# Step 4: Heterogeneous-k experiment
# ---------------------------------------------------------------------------

def run_hetero_graph(k_tok, k_cat, T, n_trials, p=P_CONN, rng=None):
    """One graph realization for the heterogeneous case.

    Two areas: token area M_tok contains assemblies A (size k_tok) and
    B (size k_tok); category area M_cat contains assembly C (size k_cat).
    A->C and B->C edges are pre-strengthened to w_0 = 2.  Training fires
    A then C, alternately B then C (the coin-flipping rule at w=2 has
    negligible effect, so this matches the baseline).

    Per-trial measurement: fire A and B both into C with fresh Gaussian
    noise.  We split the per-C-neuron noise into two independent halves,
    sigma_half = noise_std / sqrt(2), giving each side an independent
    noisy "vote".  The trial outcome is A-wins iff the total A-side
    activation across C exceeds the total B-side activation.

    Returns |P(A wins) - 0.5|.
    """
    if rng is None:
        rng = np.random.default_rng()

    # A->C and B->C edges (Bernoulli p), pre-strengthened to w0 on existing.
    W_AC = (rng.random((k_tok, k_cat)) < p).astype(np.float32) * W0
    W_BC = (rng.random((k_tok, k_cat)) < p).astype(np.float32) * W0

    # Training: balanced A->C and B->C presentations under coin-flipping.
    for _ in range(T):
        apply_cf_update(W_AC)
        apply_cf_update(W_BC)

    # Per-C-neuron total input from A and from B (all k_tok neurons fire).
    input_from_A = W_AC.sum(axis=0)   # shape (k_cat,)
    input_from_B = W_BC.sum(axis=0)   # shape (k_cat,)

    # Per-area noise: std = 5 * sqrt(k_cat * p).  Split between the two
    # competing pathways so each side gets an independent noisy vote on
    # each C neuron.  The combined effective noise per neuron has the
    # specified std = noise_std.
    noise_std = NOISE_SCALE * math.sqrt(k_cat * p)
    half_std = noise_std / math.sqrt(2.0)
    noise_a = rng.normal(0.0, half_std, size=(n_trials, k_cat)).astype(np.float32)
    noise_b = rng.normal(0.0, half_std, size=(n_trials, k_cat)).astype(np.float32)
    a_trial = input_from_A[None, :] + noise_a
    b_trial = input_from_B[None, :] + noise_b
    # Trial outcome: A wins iff total A-side activation exceeds B-side
    # across the firing C neurons (all of C fires under k-WTA on M_cat=C).
    diffs = (a_trial - b_trial).sum(axis=1)
    a_wins = int((diffs > 0).sum())
    p_A = a_wins / float(n_trials)
    return abs(p_A - 0.5)


def hetero_sweep(k_toks, k_cat, n_graphs, T, n_trials, master_seed=0):
    print(f"[hetero] plasticity_rule=coin_flipping, with Gaussian noise "
          f"(std=5*sqrt(k_cat*p) in cat area), k_cat={k_cat}, T={T} rounds, "
          f"{n_trials} trials/graph, {n_graphs} graphs/k_tok")
    results = {}
    rng_master = np.random.default_rng(master_seed)
    for k_tok in k_toks:
        t0 = time.time()
        devs = np.zeros(n_graphs, dtype=np.float32)
        for g in range(n_graphs):
            rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
            devs[g] = run_hetero_graph(k_tok, k_cat, T, n_trials, rng=rng)
        dt = time.time() - t0
        k_avg = (2 * k_tok + k_cat) / 3.0
        print(f"  k_tok={k_tok:<5d}  k_avg={k_avg:7.1f}  "
              f"mean|err|={devs.mean():.4f}  min={devs.min():.4f}  "
              f"max={devs.max():.4f}  ({dt:.1f}s)")
        results[k_tok] = devs
    return results


# ---------------------------------------------------------------------------
# Unbiased-maximum reference: expected max deviation of n_graphs Binomial
# (n_trials, 0.5) samples from 0.5.
# ---------------------------------------------------------------------------

def unbiased_max(n_trials, n_graphs, n_mc=20000, master_seed=12345):
    rng = np.random.default_rng(master_seed)
    samples = rng.binomial(n_trials, 0.5, size=(n_mc, n_graphs)) / float(n_trials)
    max_dev = np.abs(samples - 0.5).max(axis=1)
    return float(max_dev.mean())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_baseline(results, unbiased_line, out_path):
    ks = sorted(results.keys())
    means = np.array([results[k].mean() for k in ks])
    mins  = np.array([results[k].min()  for k in ks])
    maxs  = np.array([results[k].max()  for k in ks])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(ks, mins, maxs, color='C0', alpha=0.25,
                    label='min-max across 20 graphs')
    ax.plot(ks, means, color='C0', lw=2, marker='o',
            label='mean error across 20 graphs')
    ax.axhline(unbiased_line, color='k', ls='--', lw=1.0,
               label=f'unbiased max ≈ {unbiased_line:.3f}')
    ax.set_xscale('log')
    ax.set_xlabel('cap size $k$')
    ax.set_ylabel(r'error in proportion: $|P(A\ \mathrm{fires}) - 0.5|$')
    ax.set_title('Baseline (Fig. 5 reproduction): uniform $k$')
    ax.set_ylim(0, max(maxs.max(), unbiased_line) * 1.15)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hetero(baseline_results, hetero_results, k_cat, unbiased_line, out_path):
    # Uniform curve: x = k_avg = k (because in uniform case all three pieces
    # have the same k).
    ks_uniform = sorted(baseline_results.keys())
    base_means = np.array([baseline_results[k].mean() for k in ks_uniform])
    base_mins  = np.array([baseline_results[k].min()  for k in ks_uniform])
    base_maxs  = np.array([baseline_results[k].max()  for k in ks_uniform])

    k_toks = sorted(hetero_results.keys())
    k_avgs = np.array([(2 * kt + k_cat) / 3.0 for kt in k_toks])
    het_means = np.array([hetero_results[kt].mean() for kt in k_toks])
    het_mins  = np.array([hetero_results[kt].min()  for kt in k_toks])
    het_maxs  = np.array([hetero_results[kt].max()  for kt in k_toks])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(ks_uniform, base_mins, base_maxs, color='C0', alpha=0.2)
    ax.plot(ks_uniform, base_means, color='C0', lw=2, marker='o',
            label='uniform $k$ (baseline)')
    ax.fill_between(k_avgs, het_mins, het_maxs, color='C3', alpha=0.2)
    ax.plot(k_avgs, het_means, color='C3', lw=2, marker='s',
            label=r'heterogeneous: $k_{\mathrm{cat}}=500$, sweep $k_{\mathrm{tok}}$')
    ax.axhline(unbiased_line, color='k', ls='--', lw=1.0,
               label=f'unbiased max ≈ {unbiased_line:.3f}')
    ax.set_xscale('log')
    ax.set_xlabel(r'average cap size $k_{\mathrm{avg}}$')
    ax.set_ylabel(r'error in proportion: $|P(A\ \mathrm{wins}) - 0.5|$')
    ax.set_title('Heterogeneous $k$ vs uniform $k$ baseline')
    y_top = max(base_maxs.max(), het_maxs.max(), unbiased_line) * 1.15
    ax.set_ylim(0, y_top)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true',
                        help='Small/fast version (5 graphs, fewer k values)')
    args = parser.parse_args()

    if args.pilot:
        ks_baseline = [50, 100, 200, 500, 1000]
        k_toks      = [30, 50, 100, 150, 200]
        n_graphs    = 5
        n_trials    = 200
        T           = 5
    else:
        ks_baseline = [50, 100, 200, 300, 500, 750, 1000, 2000, 4000]
        k_toks      = [30, 50, 100, 150, 200]
        n_graphs    = 20
        n_trials    = 500
        T           = 10

    k_cat = 500

    print("=" * 70)
    print("STEP 3 -- Baseline (Fig. 5 reproduction)")
    print(f"  rule = coin_flipping  noise = Gaussian std=5*sqrt(k*p) on activations")
    print(f"  pre-strengthened weight w0 = {W0}")
    print(f"  cf params: alpha={CF_ALPHA}, beta={CF_BETA}, lambda={CF_LAMBDA}")
    print("=" * 70)
    baseline_results = baseline_sweep(ks_baseline, n_graphs, T, n_trials)
    unbiased_line = unbiased_max(n_trials, n_graphs)
    print(f"  unbiased-max reference (E[max dev of {n_graphs} Binomial"
          f"({n_trials},0.5)/{n_trials}]) = {unbiased_line:.4f}")
    plot_baseline(baseline_results, unbiased_line,
                  'baseline_figure5_reproduction.png')

    print()
    print("=" * 70)
    print("STEP 4 -- Heterogeneous k (novel)")
    print(f"  rule = coin_flipping  noise = Gaussian std=5*sqrt(k_cat*p) in cat area")
    print(f"  fixed k_cat = {k_cat}; sweep k_tok in {k_toks}")
    print("=" * 70)
    hetero_results = hetero_sweep(k_toks, k_cat, n_graphs, T, n_trials)
    plot_hetero(baseline_results, hetero_results, k_cat, unbiased_line,
                'heterogeneous_k_experiment.png')

    return baseline_results, hetero_results, unbiased_line


if __name__ == '__main__':
    main()
