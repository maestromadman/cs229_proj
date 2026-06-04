

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.special import erfc




N_DEFAULT = 25_000          # neurons per area
K_DEFAULT = 500             # cap size
P_DEFAULT = 0.1             # edge probability
NOISE_SCALE_DEFAULT = 5.0   # noise std = NOISE_SCALE * sqrt(k * p)

CF_ALPHA = 0.63             # plasticity ceiling
CF_BETA = 0.5               # plasticity offset
CF_LAMBDA = 26.0            # plasticity rate (Sec. 4.1); language uses 60

INTERNAL_FACTOR = 2.0       # internal/recurrent assembly weight (w_R)
BASELINE_WEIGHT = 1.0       # weiight of an un-strengthened synapse


def noise_std(k: int, p: float = P_DEFAULT, scale: float = NOISE_SCALE_DEFAULT) -> float:
    """Standard deviation of the per-neuron activation noise: scale * sqrt(k p)."""
    return scale * math.sqrt(k * p)




def coin_flip_pi(w, alpha: float = CF_ALPHA, beta: float = CF_BETA,
                 lam: float = CF_LAMBDA):
    
    arg = lam * (1.0 + beta - np.asarray(w, dtype=np.float64))
    arg = np.minimum(arg, 50.0)
    return np.minimum(alpha, np.exp(arg))


def additive_update_inplace(W: np.ndarray, alpha: float = CF_ALPHA,
                            beta: float = CF_BETA, lam: float = CF_LAMBDA) -> None:
    
    mask = W > 0
    if mask.any():
        W[mask] = W[mask] + coin_flip_pi(W[mask], alpha, beta, lam)


def train_assembly_weight(presentations: int, w0: float = BASELINE_WEIGHT,
                          alpha: float = CF_ALPHA, beta: float = CF_BETA,
                          lam: float = CF_LAMBDA) -> float:
    
    w = float(w0)
    for _ in range(presentations):
        w += float(coin_flip_pi(w, alpha, beta, lam))
    return w


def train_assembly_weights(presentations, w0: float = BASELINE_WEIGHT,
                           alpha: float = CF_ALPHA, beta: float = CF_BETA,
                           lam: float = CF_LAMBDA) -> np.ndarray:
    
    return np.array([train_assembly_weight(int(t), w0, alpha, beta, lam)
                     for t in presentations], dtype=np.float64)




@dataclass
class MeanFieldGraph:
    
    n: int
    k: int
    p: float
    m: int                             
    assembly_of: np.ndarray             
    in_degree: np.ndarray               
    assembly_neurons: list = field(default_factory=list)  


def make_meanfield_graph(n: int, k: int, p: float, m: int,
                         rng: np.random.Generator) -> MeanFieldGraph:
    
    if m * k > n:
        raise ValueError(f"m*k = {m*k} exceeds n = {n}")
    assembly_of = np.full(n, -1, dtype=np.int32)
    assembly_neurons = []
    for i in range(m):
        idx = np.arange(i * k, (i + 1) * k)
        assembly_of[idx] = i
        assembly_neurons.append(idx)
    
    in_degree = rng.binomial(k, p, size=n).astype(np.float64)
    return MeanFieldGraph(n=n, k=k, p=p, m=m, assembly_of=assembly_of,
                          in_degree=in_degree, assembly_neurons=assembly_neurons)


def _round0_counts(graph: MeanFieldGraph, input_weights: np.ndarray,
                   sigma: float, n_realizations: int, rng: np.random.Generator,
                   background_weight: float = BASELINE_WEIGHT,
                   chunk: int = 256) -> np.ndarray:
    
    n, k, m = graph.n, graph.k, graph.m
    
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
        
        top_idx = np.argpartition(total, n - k, axis=1)[:, n - k:]    
        top_assembly = graph.assembly_of[top_idx]                      
        for i in range(m):
            counts[done:done + b, i] = (top_assembly == i).sum(axis=1)
        done += b
    return counts


def _sharpen_counts(counts: np.ndarray, internal_factor: float, p: float,
                    rounds: int) -> np.ndarray:
    
    if rounds <= 0:
        return counts
    R, m = counts.shape
    k = counts.sum(axis=1).astype(np.float64)            
    c = counts.astype(np.float64)
    for _ in range(rounds):
        F = c.sum(axis=1)                                
        mu = p * (F[:, None] + (internal_factor - 1.0) * c)
        var = p * (1.0 - p) * (internal_factor ** 2 * c + (F[:, None] - c))
        sigma = np.sqrt(np.maximum(var, 1e-9))
        
        lo = (mu - 6 * sigma).min(axis=1)
        hi = (mu + 6 * sigma).max(axis=1)
        for _bisect in range(30):
            tau = 0.5 * (lo + hi)
            
            z = (tau[:, None] - mu) / sigma
            above = 0.5 * erfc(z / math.sqrt(2.0)) * k[:, None]   
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
    
    input_weights = np.asarray(input_weights, dtype=np.float64)
    counts = _round0_counts(graph, input_weights, sigma, n_realizations, rng,
                            background_weight=background_weight)
    counts = _sharpen_counts(counts, internal_factor, graph.p, recurrent_rounds)
    
    jitter = rng.random(counts.shape)
    return np.argmax(counts + jitter, axis=1).astype(np.int32)


def win_probability(graph: MeanFieldGraph, input_weights, sigma: float,
                    n_realizations: int, rng: np.random.Generator,
                    target: int = 0, **kwargs) -> float:
    """Empirical Pr(assembly `target` wins) over `n_realizations` noise draws."""
    winners = sample_winners_meanfield(graph, input_weights, sigma,
                                       n_realizations, rng, **kwargs)
    return float((winners == target).mean())




@dataclass
class SparseArea:
    
    n: int
    k: int
    p: float
    m: int
    assembly_of: np.ndarray
    assembly_neurons: list
    W: sp.csr_matrix          
    in_degree: np.ndarray     


def build_sparse_area(n: int, k: int, p: float, m: int,
                      rng: np.random.Generator,
                      internal_factor: float = INTERNAL_FACTOR) -> SparseArea:
   
    assembly_of = np.full(n, -1, dtype=np.int32)
    assembly_neurons = []
    for i in range(m):
        idx = np.arange(i * k, (i + 1) * k)
        assembly_of[idx] = i
        assembly_neurons.append(idx)

    
    W = sp.random(n, n, density=p, format='csr', dtype=np.float32,
                  random_state=rng, data_rvs=np.ones)
    W.setdiag(0)
    W.eliminate_zeros()
    
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
        return int(rng.integers(m))          
    best = np.flatnonzero(counts == counts.max())
    return int(rng.choice(best))




def torch_device(prefer="auto"):
    """Resolve a torch device.  'auto' -> cuda if present else cpu."""
    import torch
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def build_dense_area_torch(n, k, p, m, device, *,
                           internal_factor=INTERNAL_FACTOR, dtype=None,
                           np_rng=None, torch_gen=None):
    
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



def softmax_weights(weights, lam: float) -> np.ndarray:
    
    w = np.asarray(weights, dtype=np.float64)
    z = lam * (w - w.max())
    e = np.exp(z)
    return e / e.sum()


def fit_softmax_lambda(weights_matrix, empirical, lam_grid=None):
    
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
    
    if rng is None:
        rng = np.random.default_rng(12345)
    samples = rng.binomial(n_trials, 0.5, size=(n_mc, n_graphs)) / float(n_trials)
    return float(np.abs(samples - 0.5).max(axis=1).mean())




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
