"""Extensions to the Assembly Calculus / NEMO model for the experiments in

    Dabagia, Mitropolsky, Papadimitriou & Vempala (2024),
    "Coin-Flipping In The Brain: Statistical Learning with Neuronal Assemblies"
    arXiv:2406.07715

This module provides the two model ingredients that differ from the stock
multiplicative-plasticity NEMO in the surrounding repo:

  1. The *additive* coin-flipping plasticity rule
         pi(w) = min(alpha, exp(lambda * (1 + beta - w)))
         w_ij(t+1) = w_ij(t) + pi(w_ij(t))      (only when both endpoints fire)
     with the constraint pi(0) = 0 so non-existent synapses are never created.

  2. A *noisy k-cap* firing step
         x_A(t+1) = k-cap( sum_B W_{B,A} x_B(t) + z_A(t) ),  z ~ N(0, sigma^2 k p).

On top of these it offers two simulation "engines" for the assembly coin-flip:

  * MEAN-FIELD engine (fast, the default).  Assemblies are taken as fixed
    neuron sets (as the paper does: "we take assemblies as a starting point").
    A context assembly I fires once into a state area S; each S neuron's input
    is its (graph-fixed) number of synapses from I, scaled by the per-assembly
    "weight", plus i.i.d. Gaussian noise.  The k-cap picks the round-0 firing
    set; the recurrent attractor dynamics (internal weights strengthened by a
    factor `internal_factor`) then drive S to whichever assembly has the
    plurality of round-0 firers.  Because that recurrent map is monotone in the
    per-assembly firer counts, the converged winner equals the round-0 argmax,
    which is what we read off.  This makes paper-scale repetition counts
    (100 graphs x 1000 noise realizations) run in minutes.

  * SPARSE engine (faithful, slow, behind a flag).  Builds the real n x n
    recurrent connectome with scipy.sparse and runs true noisy k-cap rounds to
    convergence.  Intended for cross-validation at modest scale / on bigger
    compute; not run by default.

All randomness is routed through an explicit numpy Generator for reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.special import erfc


# --------------------------------------------------------------------------- #
# Global parameters (Dabagia et al. 2024, Section 4)                          #
# --------------------------------------------------------------------------- #

N_DEFAULT = 25_000          # neurons per area
K_DEFAULT = 500             # cap size
P_DEFAULT = 0.1             # edge probability
NOISE_SCALE_DEFAULT = 5.0   # noise std = NOISE_SCALE * sqrt(k * p)

CF_ALPHA = 0.63             # plasticity ceiling
CF_BETA = 0.5               # plasticity offset
CF_LAMBDA = 26.0            # plasticity rate (Sec. 4.1); language uses 60

INTERNAL_FACTOR = 2.0       # internal/recurrent assembly weight (w_R), "factor of 2"
BASELINE_WEIGHT = 1.0       # weight of an un-strengthened synapse


def noise_std(k: int, p: float = P_DEFAULT, scale: float = NOISE_SCALE_DEFAULT) -> float:
    """Standard deviation of the per-neuron activation noise: scale * sqrt(k p)."""
    return scale * math.sqrt(k * p)


# --------------------------------------------------------------------------- #
# Plasticity rule                                                             #
# --------------------------------------------------------------------------- #

def coin_flip_pi(w, alpha: float = CF_ALPHA, beta: float = CF_BETA,
                 lam: float = CF_LAMBDA):
    """Additive plasticity increment pi(w) = min(alpha, exp(lam (1 + beta - w))).

    Works on scalars or numpy arrays.  The exponent is clipped before exp() to
    avoid float overflow warnings: once lam(1+beta-w) exceeds ~log(alpha) the
    min() returns alpha regardless, so clipping the argument is exact.
    """
    arg = lam * (1.0 + beta - np.asarray(w, dtype=np.float64))
    arg = np.minimum(arg, 50.0)
    return np.minimum(alpha, np.exp(arg))


def additive_update_inplace(W: np.ndarray, alpha: float = CF_ALPHA,
                            beta: float = CF_BETA, lam: float = CF_LAMBDA) -> None:
    """Apply w <- w + pi(w) in place, but ONLY on existing synapses (w > 0).

    Enforces pi(0) = 0: the additive rule would otherwise manufacture synapses
    on every coincident pair of firings (since pi(0) = alpha > 0).
    """
    mask = W > 0
    if mask.any():
        W[mask] = W[mask] + coin_flip_pi(W[mask], alpha, beta, lam)


def train_assembly_weight(presentations: int, w0: float = BASELINE_WEIGHT,
                          alpha: float = CF_ALPHA, beta: float = CF_BETA,
                          lam: float = CF_LAMBDA) -> float:
    """Scalar weight after `presentations` Hebbian updates of one assembly.

    Every synapse from the context assembly I into outcome assembly A_i starts
    at the same weight w0 and receives the same increment pi(w) on each of the
    `presentations` times A_i is fired after I, so the common edge weight stays
    uniform and the whole "assembly weight" follows the scalar recurrence
    w <- w + pi(w).  Order across assemblies is irrelevant (each assembly's
    edges are only touched when that assembly is presented).
    """
    w = float(w0)
    for _ in range(presentations):
        w += float(coin_flip_pi(w, alpha, beta, lam))
    return w


def train_assembly_weights(presentations, w0: float = BASELINE_WEIGHT,
                           alpha: float = CF_ALPHA, beta: float = CF_BETA,
                           lam: float = CF_LAMBDA) -> np.ndarray:
    """Vectorized `train_assembly_weight` over a list of presentation counts."""
    return np.array([train_assembly_weight(int(t), w0, alpha, beta, lam)
                     for t in presentations], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Mean-field engine                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class MeanFieldGraph:
    """One random-graph realization for the mean-field engine.

    Stores only what the engine needs: which assembly each neuron belongs to
    (-1 for background) and each neuron's number of synapses from the context
    assembly I (Binomial(k, p)), held FIXED so that graph-to-graph asymmetry is
    a genuine, reproducible source of variance across trials.
    """
    n: int
    k: int
    p: float
    m: int                              # number of outcome assemblies
    assembly_of: np.ndarray             # (n,) int, values in {-1, 0..m-1}
    in_degree: np.ndarray               # (n,) int, synapses from I
    assembly_neurons: list = field(default_factory=list)  # list of index arrays


def make_meanfield_graph(n: int, k: int, p: float, m: int,
                         rng: np.random.Generator) -> MeanFieldGraph:
    """Sample one mean-field graph: assign m disjoint assemblies of size k and
    draw each neuron's in-degree from the (k-neuron) context assembly I."""
    if m * k > n:
        raise ValueError(f"m*k = {m*k} exceeds n = {n}")
    assembly_of = np.full(n, -1, dtype=np.int32)
    assembly_neurons = []
    for i in range(m):
        idx = np.arange(i * k, (i + 1) * k)
        assembly_of[idx] = i
        assembly_neurons.append(idx)
    # Number of synapses from the k firing neurons of I into each S neuron.
    in_degree = rng.binomial(k, p, size=n).astype(np.float64)
    return MeanFieldGraph(n=n, k=k, p=p, m=m, assembly_of=assembly_of,
                          in_degree=in_degree, assembly_neurons=assembly_neurons)


def _round0_counts(graph: MeanFieldGraph, input_weights: np.ndarray,
                   sigma: float, n_realizations: int, rng: np.random.Generator,
                   background_weight: float = BASELINE_WEIGHT,
                   chunk: int = 256) -> np.ndarray:
    """Per-assembly firer counts after the single noisy I-driven k-cap round.

    Returns an (n_realizations, m) integer array: how many neurons of each
    outcome assembly land in the top-k when I fires once with Gaussian noise.
    """
    n, k, m = graph.n, graph.k, graph.m
    # Per-neuron deterministic input from I: weight(assembly) * in_degree.
    weight_per_neuron = np.full(n, background_weight, dtype=np.float64)
    for i in range(m):
        weight_per_neuron[graph.assembly_neurons[i]] = input_weights[i]
    input_mean = weight_per_neuron * graph.in_degree

    counts = np.empty((n_realizations, m), dtype=np.int32)
    done = 0
    while done < n_realizations:
        b = min(chunk, n_realizations - done)
        noise = rng.normal(0.0, sigma, size=(b, n))
        total = input_mean[None, :] + noise
        # Indices of the top-k inputs for each realization.
        top_idx = np.argpartition(total, n - k, axis=1)[:, n - k:]    # (b, k)
        top_assembly = graph.assembly_of[top_idx]                      # (b, k)
        for i in range(m):
            counts[done:done + b, i] = (top_assembly == i).sum(axis=1)
        done += b
    return counts


def _sharpen_counts(counts: np.ndarray, internal_factor: float, p: float,
                    rounds: int) -> np.ndarray:
    """Optional deterministic recurrent sharpening of the per-assembly counts.

    Models subsequent (noise-free) rounds in which only S fires: an assembly-i
    neuron's expected recurrent input is p*(F + (internal_factor-1) c_i), so
    higher-count assemblies pull more of the cap each round.  The map is
    monotone in the counts, so it never changes the argmax (the converged
    winner) -- it is provided only to expose the attractor explicitly and to
    mirror the sparse engine.  `rounds=0` (the default for sampling) is
    therefore equivalent for winner determination.
    """
    if rounds <= 0:
        return counts
    R, m = counts.shape
    k = counts.sum(axis=1).astype(np.float64)            # cap size per realization
    c = counts.astype(np.float64)
    for _ in range(rounds):
        F = c.sum(axis=1)                                # current firers
        mu = p * (F[:, None] + (internal_factor - 1.0) * c)
        var = p * (1.0 - p) * (internal_factor ** 2 * c + (F[:, None] - c))
        sigma = np.sqrt(np.maximum(var, 1e-9))
        # Find per-realization threshold tau s.t. sum_i k_i * P(N(mu_i,sig_i)>tau)=k
        lo = (mu - 6 * sigma).min(axis=1)
        hi = (mu + 6 * sigma).max(axis=1)
        for _bisect in range(30):
            tau = 0.5 * (lo + hi)
            # expected number of assembly neurons above tau (each assembly has k neurons)
            z = (tau[:, None] - mu) / sigma
            above = 0.5 * erfc(z / math.sqrt(2.0)) * k[:, None]   # treat each assembly pool size = k? use counts?
            tot = above.sum(axis=1)
            himask = tot > k
            lo = np.where(himask, tau, lo)
            hi = np.where(himask, hi, tau)
        z = (tau[:, None] - mu) / sigma
        c = 0.5 * erfc(z / math.sqrt(2.0)) * k[:, None]
    return c


def sample_winners_meanfield(graph: MeanFieldGraph, input_weights,
                             sigma: float, n_realizations: int,
                             rng: np.random.Generator,
                             internal_factor: float = INTERNAL_FACTOR,
                             recurrent_rounds: int = 0,
                             background_weight: float = BASELINE_WEIGHT
                             ) -> np.ndarray:
    """Winning-assembly index for each noise realization (mean-field engine).

    Returns an (n_realizations,) int array of values in {0..m-1}.  Ties in the
    round-0 counts are broken uniformly at random.
    """
    input_weights = np.asarray(input_weights, dtype=np.float64)
    counts = _round0_counts(graph, input_weights, sigma, n_realizations, rng,
                            background_weight=background_weight)
    counts = _sharpen_counts(counts, internal_factor, graph.p, recurrent_rounds)
    # Random tie-break: add jitter in [0,1) -- preserves strict ordering of
    # integer counts, randomizes exact ties.
    jitter = rng.random(counts.shape)
    return np.argmax(counts + jitter, axis=1).astype(np.int32)


def win_probability(graph: MeanFieldGraph, input_weights, sigma: float,
                    n_realizations: int, rng: np.random.Generator,
                    target: int = 0, **kwargs) -> float:
    """Empirical Pr(assembly `target` wins) over `n_realizations` noise draws."""
    winners = sample_winners_meanfield(graph, input_weights, sigma,
                                       n_realizations, rng, **kwargs)
    return float((winners == target).mean())


# --------------------------------------------------------------------------- #
# Sparse engine (faithful, slow -- behind a flag, not run by default)         #
# --------------------------------------------------------------------------- #

@dataclass
class SparseArea:
    """A fully-instantiated state area S for the faithful engine."""
    n: int
    k: int
    p: float
    m: int
    assembly_of: np.ndarray
    assembly_neurons: list
    W: sp.csr_matrix          # n x n recurrent weights (row=source, col=target)
    in_degree: np.ndarray     # synapses from context assembly I


def build_sparse_area(n: int, k: int, p: float, m: int,
                      rng: np.random.Generator,
                      internal_factor: float = INTERNAL_FACTOR) -> SparseArea:
    """Instantiate the real n x n recurrent connectome (Erdos-Renyi(p)), with
    the within-assembly synapses strengthened by `internal_factor`.

    WARNING: at n=25000, p=0.1 this is ~62M nonzeros (~0.7-1 GB as CSR).  Build
    one graph at a time and discard before the next.  This path is intended for
    cross-validation at modest n, or for large runs on dedicated compute.
    """
    assembly_of = np.full(n, -1, dtype=np.int32)
    assembly_neurons = []
    for i in range(m):
        idx = np.arange(i * k, (i + 1) * k)
        assembly_of[idx] = i
        assembly_neurons.append(idx)

    # Sparse Erdos-Renyi adjacency with weight 1.
    W = sp.random(n, n, density=p, format='csr', dtype=np.float32,
                  random_state=rng, data_rvs=np.ones)
    W.setdiag(0)
    W.eliminate_zeros()
    # Strengthen within-assembly synapses by internal_factor.  Assemblies are
    # contiguous blocks, so an edge is internal iff its row and column belong to
    # the same assembly; scale W.data in place via the CSR structure (no tolil).
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(W.indptr))
    cols = W.indices
    same = (assembly_of[rows] == assembly_of[cols]) & (assembly_of[rows] >= 0)
    W.data[same] *= internal_factor

    in_degree = rng.binomial(k, p, size=n).astype(np.float32)
    return SparseArea(n=n, k=k, p=p, m=m, assembly_of=assembly_of,
                      assembly_neurons=assembly_neurons, W=W, in_degree=in_degree)


def _top_k_indicator(values: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(values, len(values) - k)[len(values) - k:]
    x = np.zeros(len(values), dtype=np.float32)
    x[idx] = 1.0
    return x


def sample_winner_sparse(area: SparseArea, input_weights, sigma: float,
                         rng: np.random.Generator, max_rounds: int = 30,
                         background_weight: float = BASELINE_WEIGHT,
                         noise_every_round: bool = False) -> int:
    """One faithful noisy-k-cap convergence run; returns the winning assembly.

    Round 0: I fires once -> per-neuron input = weight(assembly)*in_degree, plus
    N(0, sigma^2) noise; take the k-cap.  Rounds 1..: only S fires (recurrent
    W^T x); noise off by default (paper: "in subsequent rounds, only S fires").
    Convergence = identical firing set on two consecutive rounds.  The winner is
    the assembly holding the plurality of the converged cap.
    """
    n, k, m = area.n, area.k, area.m
    weight_per_neuron = np.full(n, background_weight, dtype=np.float32)
    for i in range(m):
        weight_per_neuron[area.assembly_neurons[i]] = input_weights[i]
    input_mean = weight_per_neuron * area.in_degree

    total = input_mean + rng.normal(0.0, sigma, size=n).astype(np.float32)
    x = _top_k_indicator(total, k)
    prev = None
    for _ in range(max_rounds):
        rec = area.W.T.dot(x)
        if noise_every_round:
            rec = rec + rng.normal(0.0, sigma, size=n).astype(np.float32)
        x = _top_k_indicator(rec, k)
        cur = np.flatnonzero(x)
        if prev is not None and len(cur) == len(prev) and np.array_equal(cur, prev):
            break
        prev = cur
    firing = np.flatnonzero(x)
    assemblies = area.assembly_of[firing]
    counts = np.array([(assemblies == i).sum() for i in range(m)])
    if counts.max() == 0:
        return int(rng.integers(m))          # no assembly captured the cap
    best = np.flatnonzero(counts == counts.max())
    return int(rng.choice(best))


# --------------------------------------------------------------------------- #
# Dense batched GPU engine (PyTorch) -- faithful, fast on a GPU                #
# --------------------------------------------------------------------------- #
#
# Same model as the sparse engine, but the recurrent connectome is a DENSE
# tensor and the noise realizations are processed as a batch, so each recurrent
# round is one big matmul (x @ W).  At n=25000 a float32 dense matrix is ~2.5 GB,
# which fits a 24 GB GPU (e.g. an NVIDIA L4) with room to spare; the whole Fig. 5
# sweep then runs in minutes instead of hours.  torch is an OPTIONAL dependency
# (imported lazily) so the other experiments work without it.

def torch_device(prefer="auto"):
    """Resolve a torch device.  'auto' -> cuda if present else cpu."""
    import torch
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def build_dense_area_torch(n, k, p, m, device, *,
                           internal_factor=INTERNAL_FACTOR, dtype=None,
                           np_rng=None, torch_gen=None):
    """Instantiate the n x n recurrent connectome as a DENSE tensor on `device`.

    Erdos-Renyi(p) adjacency with weight 1, within-assembly blocks scaled by
    `internal_factor`, self-loops removed.  in_degree (synapses from the context
    assembly I) is drawn with the numpy generator for reproducibility and moved
    to the device.  Returns (W, in_degree, assembly_of) tensors on `device`.
    """
    import torch
    if dtype is None:
        dtype = torch.float32
    if np_rng is None:
        np_rng = np.random.default_rng()
    W = (torch.rand(n, n, device=device, generator=torch_gen, dtype=dtype) < p).to(dtype)
    W.fill_diagonal_(0.0)
    for i in range(m):
        s = slice(i * k, (i + 1) * k)
        W[s, s] *= internal_factor
    in_degree = torch.as_tensor(np_rng.binomial(k, p, size=n), device=device, dtype=dtype)
    assembly_of = torch.full((n,), -1, dtype=torch.long, device=device)
    for i in range(m):
        assembly_of[i * k:(i + 1) * k] = i
    return W, in_degree, assembly_of


def sample_winners_torch(W, in_degree, assembly_of, input_weights, k, sigma,
                         n_samples, *, max_rounds=20,
                         background_weight=BASELINE_WEIGHT,
                         noise_every_round=False, batch=None, torch_gen=None):
    """Faithful batched noisy-k-cap convergence; winner index per noise sample.

    Round 0: I fires once -> per-neuron input = weight(assembly)*in_degree plus
    N(0, sigma^2); take the k-cap.  Rounds 1..max_rounds: only S fires, recurrent
    input is x @ W (x the 0/1 firing-set row), noise off by default (paper:
    "in subsequent rounds, only S fires").  Winner = assembly with the plurality
    of the converged cap (random tie-break).  Noise samples are processed in
    batches of `batch` rows to bound memory.
    """
    import torch
    device = W.device
    n = W.shape[0]
    m = len(input_weights)
    wpn = torch.full((n,), float(background_weight), device=device, dtype=W.dtype)
    for i in range(m):
        wpn[assembly_of == i] = float(input_weights[i])
    input_mean = wpn * in_degree
    if batch is None:
        batch = n_samples
    winners = torch.empty(n_samples, dtype=torch.long, device=device)
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        total = input_mean.unsqueeze(0) + sigma * torch.randn(
            b, n, device=device, generator=torch_gen, dtype=W.dtype)
        idx = torch.topk(total, k, dim=1).indices            # (b, k)
        for _ in range(max_rounds):
            X = torch.zeros(b, n, device=device, dtype=W.dtype)
            X.scatter_(1, idx, 1.0)
            rec = X @ W                                       # (b, n)
            if noise_every_round:
                rec = rec + sigma * torch.randn(
                    b, n, device=device, generator=torch_gen, dtype=W.dtype)
            idx = torch.topk(rec, k, dim=1).indices
        asm = assembly_of[idx]                                # (b, k)
        counts = torch.stack([(asm == i).sum(dim=1) for i in range(m)],
                             dim=1).to(torch.float32)
        jitter = torch.rand(b, m, device=device, generator=torch_gen)
        winners[done:done + b] = torch.argmax(counts + jitter, dim=1)
        done += b
    return winners


# --------------------------------------------------------------------------- #
# Reference statistics & helpers                                              #
# --------------------------------------------------------------------------- #

def softmax_weights(weights, lam: float) -> np.ndarray:
    """softmax(lam * weights) -- the paper's target distribution over outcomes."""
    w = np.asarray(weights, dtype=np.float64)
    z = lam * (w - w.max())
    e = np.exp(z)
    return e / e.sum()


def fit_softmax_lambda(weights_matrix, empirical, lam_grid=None):
    """Choose lambda minimizing MSE between softmax([w_i])[0] and `empirical`.

    `weights_matrix`: (num_points, m) array of the weight vectors swept.
    `empirical`:      (num_points,) measured Pr(A1 wins).
    Returns (best_lambda, fitted_curve).
    """
    weights_matrix = np.asarray(weights_matrix, dtype=np.float64)
    empirical = np.asarray(empirical, dtype=np.float64)
    if lam_grid is None:
        lam_grid = np.linspace(0.5, 60.0, 600)
    best_lam, best_mse, best_curve = None, np.inf, None
    for lam in lam_grid:
        curve = np.array([softmax_weights(w, lam)[0] for w in weights_matrix])
        mse = float(np.mean((curve - empirical) ** 2))
        if mse < best_mse:
            best_lam, best_mse, best_curve = lam, mse, curve
    return best_lam, best_curve


def unbiased_max_deviation(n_trials: int, n_graphs: int, n_mc: int = 20000,
                           rng: np.random.Generator | None = None) -> float:
    """E[ max over `n_graphs` of |Binomial(n_trials,1/2)/n_trials - 1/2| ].

    The "Unbiased Maximum" reference in Fig. 5: the largest deviation one would
    expect to see across the trials if every learned distribution were exactly
    1/2 and all error were pure sampling noise.
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    samples = rng.binomial(n_trials, 0.5, size=(n_mc, n_graphs)) / float(n_trials)
    return float(np.abs(samples - 0.5).max(axis=1).mean())


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sigma = noise_std(K_DEFAULT)
    print(f"noise std @ k={K_DEFAULT}: {sigma:.2f}")
    print("pi(1.0) =", coin_flip_pi(1.0), " pi(2.0) =", coin_flip_pi(2.0))
    w_trained = train_assembly_weights([1, 5, 10, 20, 40])
    print("trained weights (T=1,5,10,20,40):", np.round(w_trained, 4))

    g = make_meanfield_graph(N_DEFAULT, K_DEFAULT, P_DEFAULT, m=3, rng=rng)
    for w1 in (1.3, 1.5, 1.7):
        pA = win_probability(g, [w1, 1.5, 1.5], sigma, 400, rng, target=0)
        print(f"  w1={w1}: Pr(A1 wins) ~ {pA:.3f}")
    print("unbiased max dev (500 trials, 20 graphs):",
          round(unbiased_max_deviation(500, 20), 4))
