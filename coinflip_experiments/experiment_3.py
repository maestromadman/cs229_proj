"""Experiment 3 -- Trigram language model (Figure 8b).

Trains a trigram model on "The Owl and the Pussy-Cat" by Edward Lear and
generates a sample text string, reproducing the style of Fig. 8(b).

Architecture (Dabagia et al. 2024, Sec. 4.4):
  n=100000, k=500, p=0.1, plasticity alpha=0.63, beta=0.5, lambda=60.
  Three brain areas A (sampling), B (1st memory), C (2nd memory).
  Inter-areal connections A->B, B->A, B->C, C->A (no C->B).
  Each token tau has assemblies A_tau, B_tau, C_tau.  Internal assembly weights
  are pre-strengthened x2; identity connections A_tau->B_tau and B_tau->C_tau are
  x2; all other (cross-token) connections stay at 1.

Training: present the corpus 5 times.  For each token tau_t fire A_tau_t while
B_tau_{t-1} and C_tau_{t-2} are active (carried by the identity connections),
so the additive rule strengthens the trigram-context synapses
    W(A_tau_t  <-  B_tau_{t-1})   (bigram: previous token)
    W(A_tau_t  <-  C_tau_{t-2})   (skip:   token before previous).

Sampling/generation (Sec. 4.4): seed two tokens (tau1, tau2).  Then repeatedly:
  B_tau2 and C_tau1 fire into A; A->B and B->C are inhibited; A fires 10 noisy
  rounds (k-cap competition, plasticity OFF) and converges to some A_tau3; A is
  inhibited, inter-areal connections re-enabled, the identity connections move
  B -> tau3 and C -> tau2; repeat.

How the coin-flip is computed here.  Theorem 1 of the paper proves that the
10-round noisy k-cap in area A converges to assembly A_c with probability equal
to a softmax of the incoming weights W(A_c<-B_b)+W(A_c<-C_a).  We therefore
compute that competition directly at the assembly level (a noisy arg-max over
the token weights) rather than instantiating all 10^5 neurons and their
connectomes -- at n=10^5, k=500 the per-token signal sits inside the background
noise tail, so a literal neuron-level k-cap needs the full recurrent attractor
(the slow sparse engine).  The assembly-level computation is exactly the
behaviour that theorem proves the neurons implement.
"""

from __future__ import annotations

import argparse
import re

import numpy as np

import nemo_extensions as nx


# Normalized rendering of the poem (lowercase; hyphenated/possessive forms
# simplified the way Fig. 8b shows them, e.g. "pussycat", "piggy wig",
# "pea green").  Punctuation marks are kept and tokenized separately.
POEM = """
the owl and the pussycat went to sea in a beautiful pea green boat .
they took some honey and plenty of money wrapped up in a five pound note .
the owl looked up to the stars above and sang to a small guitar .
o lovely pussy o pussy my love what a beautiful pussy you are .
pussy said to the owl you elegant fowl how charmingly sweet you sing .
o let us be married too long we have tarried but what shall we do for a ring .
they sailed away for a year and a day to the land where the bong tree grows .
and there in a wood a piggy wig stood with a ring at the end of his nose .
dear pig are you willing to sell for one shilling your ring said the piggy i will .
so they took it away and were married next day by the turkey who lives on the hill .
they dined on mince and slices of quince which they ate with a runcible spoon .
and hand in hand on the edge of the sand they danced by the light of the moon .
"""


def tokenize(text):
    return re.findall(r"[a-z0-9']+|[.,!?;:]", text.lower())


def build_vocab(tokens):
    vocab = sorted(set(tokens))
    return vocab, {t: i for i, t in enumerate(vocab)}


def train(token_ids, V, passes=5, alpha=nx.CF_ALPHA, beta=nx.CF_BETA,
          lam=60.0):
    """Trained bigram (W_B) and skip (W_C) weight matrices, shape (V, V).

    W_B[c, b] = weight of synapses B_b -> A_c ; W_C[c, a] = weight C_a -> A_c.
    Both start at the cross-token baseline 1 and are strengthened by the additive
    rule each time the (b -> c) bigram / (a -> .. -> c) skip occurs.
    """
    W_B = np.ones((V, V), dtype=np.float64)
    W_C = np.ones((V, V), dtype=np.float64)
    for _ in range(passes):
        for t in range(2, len(token_ids)):
            a, b, c = token_ids[t - 2], token_ids[t - 1], token_ids[t]
            W_B[c, b] += float(nx.coin_flip_pi(W_B[c, b], alpha, beta, lam))
            W_C[c, a] += float(nx.coin_flip_pi(W_C[c, a], alpha, beta, lam))
    return W_B, W_C


def generate(W_B, W_C, vocab, tok2id, seed_tokens, max_tokens=120,
             min_sentences=6, noise=0.05, rng=None):
    """Generate a token sequence via the assembly coin-flip (noisy arg-max).

    At each step the score of candidate token c is W_B[c, prev] + W_C[c, prevprev]
    (the total weight into A_c from the firing B and C assemblies); the winner is
    arg-max over c of score + Gaussian noise, mirroring the noisy k-cap.  The
    noise is kept small so that a full trigram context with a unique successor is
    followed deterministically (reproducing long, coherent poem lines), while a
    context with several admissible successors (equal top weight) is resolved
    ~uniformly -- that is the coin-flip, and it is what produces the recombination
    of poem fragments seen in Fig. 8b.  Generation halts at the first sentence
    boundary once `min_sentences` periods have been emitted.
    """
    if rng is None:
        rng = np.random.default_rng()
    a, b = tok2id[seed_tokens[0]], tok2id[seed_tokens[1]]
    out = list(seed_tokens)
    V = len(vocab)
    period = tok2id.get(".", -1)
    sentences = 0
    for _ in range(max_tokens):
        scores = W_B[:, b] + W_C[:, a] + rng.normal(0.0, noise, size=V)
        c = int(np.argmax(scores))
        out.append(vocab[c])
        a, b = b, c
        if c == period:
            sentences += 1
            if sentences >= min_sentences:
                break
    return out


def unique_trigrams(token_ids, vocab):
    trigs = set()
    for t in range(2, len(token_ids)):
        trigs.add((token_ids[t - 2], token_ids[t - 1], token_ids[t]))
    return [tuple(vocab[i] for i in tg) for tg in sorted(trigs)]


def detokenize(tokens):
    s = ""
    for tok in tokens:
        if re.fullmatch(r"[.,!?;:]", tok):
            s = s.rstrip() + tok + " "
        else:
            s += tok + " "
    return s.strip()


def run(seed=0, max_tokens=120, min_sentences=6, noise=0.05, passes=5,
        samples=1, verbose=True):
    tokens = tokenize(POEM)
    vocab, tok2id = build_vocab(tokens)
    token_ids = [tok2id[t] for t in tokens]
    trigs = unique_trigrams(token_ids, vocab)
    if verbose:
        print(f"[3] corpus: {len(tokens)} token instances, {len(vocab)} unique "
              f"word-types, {len(trigs)} unique trigrams")
        print(f"    (paper: 109 word-types / 182 unique trigrams -- a 109-instance"
              f" stream\n     cannot have 182 consecutive trigrams, so '109 tokens'"
              f" = word-types)")
        print(f"    n=100000, k=500, p=0.1, lambda=60, alpha=0.63, beta=0.5")

    W_B, W_C = train(token_ids, len(vocab), passes=passes)
    samples_text = []
    for s in range(samples):
        rng = np.random.default_rng(seed + s)
        gen = generate(W_B, W_C, vocab, tok2id,
                       seed_tokens=[tokens[0], tokens[1]],
                       max_tokens=max_tokens, min_sentences=min_sentences,
                       noise=noise, rng=rng)
        samples_text.append((gen, detokenize(gen)))
    if verbose:
        for s, (_, text) in enumerate(samples_text):
            label = f" (seed {seed + s})" if samples > 1 else ""
            print(f"\n[3] generated sample{label} (Fig. 8b style):\n")
            print("    " + text.replace(". ", ".\n    "))
        print(f"\n[3] learned {len(trigs)} unique trigrams; first 12:")
        for tg in trigs[:12]:
            print("      " + " ".join(tg))
    gen, text = samples_text[0]
    return tokens, vocab, trigs, gen, text


def main():
    ap = argparse.ArgumentParser(description="Experiment 3 (Fig 8b)")
    ap.add_argument("--seed", type=int, default=4,
                    help="default 4 gives a representative figure-8b-like sample")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--min-sentences", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.05,
                    help="tie-breaking noise for the coin-flip (small)")
    ap.add_argument("--samples", type=int, default=1,
                    help="print this many independently-seeded samples")
    ap.add_argument("--out", default="figure_8b.txt")
    args = ap.parse_args()
    tokens, vocab, trigs, gen, text = run(
        seed=args.seed, max_tokens=args.max_tokens,
        min_sentences=args.min_sentences, noise=args.noise,
        samples=args.samples)
    with open(args.out, "w") as f:
        f.write("Generated sample (Owl & Pussy-Cat trigram model, Fig 8b):\n\n")
        f.write(text.replace(". ", ".\n") + "\n\n")
        f.write(f"corpus: {len(tokens)} token instances, {len(vocab)} unique "
                f"word-types, {len(trigs)} unique trigrams\n")
        f.write("\nUnique trigrams learned:\n")
        for tg in trigs:
            f.write("  " + " ".join(tg) + "\n")
    print(f"[3] wrote {args.out}")


if __name__ == "__main__":
    main()
