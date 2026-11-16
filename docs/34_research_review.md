# 34 · Deep research review (six agents, one night)

Six independent research agents ran against the arXiv API (DuckDuckGo and Wikipedia are blocked
in this environment). Every number below was taken from a fetched paper. This file records what
they found, what it says about CortexMesh, and the order in which to act.

## The one-line diagnosis

Our memory saturates because we *reinforce the same pair forever*. A positive-only accumulator
with a `(1 - w)` ceiling sends every co-active pair to 1; after enough tokens every pair is 1 and
the read-out has nothing left to discriminate. The literature's answer is not a better ceiling, it
is a *different state update*: subtract the mean, gate with a delta rule, or replace the matrix
with a bounded cache.

## Finding 1 · Dense associative memory (agent: Hopfield capacity)

- Classical Hopfield capacity is `0.138 N`; dense associative memory raises it to `alpha_n N^(n-1)`
  by applying the non-linearity `F` to the **overlap with each stored pattern**, never to
  individual weights (Krotov & Hopfield 2016, arXiv:1606.01164).
- Modern Hopfield read-out is `xi_new = X softmax(beta X^T xi)` — a softmax *over patterns*, i.e.
  a competition between candidates, not a sum of synapses (Ramsauer et al. 2020, arXiv:2008.02217).
- Willshaw density: `p1 = 1 - (1 - f^2)^M`. With our `f_in = 9.8e-4`, `f_out = 2.3e-2`, saturation
  onset is `M* = 1/(f_in f_out) = 4.4e4` patterns; at 8e6 patterns each weight expects ~183 hits
  and `p1 = 1 - e^-183 ≈ 1`. **Our 96.9% is exactly the predicted Willshaw density.**
- Consequence: a positive-only accumulator cannot be fixed by tuning. Counts converge to a
  constant matrix plus noise.

## Finding 2 · Delta rule and gated linear attention (agent: fast weights)

- DeltaNet: `S_t = S_{t-1}(I - beta k k^T) + beta v k^T`. Error-proportional, no ceiling, state
  size `d_k x d_v`, ~`d_k` orthogonal keys.
- Gated variants bound the state at `beta||v||/(1-a)` at the cost of a biased fixed point.
- Measured: DeltaNet 1.3B 16.87/12.21 ppl (WikiText/C4) vs Mamba 17.06/13.89, GLA 17.22/14.47.
- Selective state (S-NIAH, 4k context): DeltaNet 18.6, Mamba-2 56.2, GDN 92.2 — pure delta
  collides, pure decay forgets.
- Our rule is the pure Hebbian accumulator (no `-k k^T` term), which is why it has no fixed point
  below 1.

## Finding 3 · Continual learning and loss of plasticity (agent: anti-forgetting)

- EWC `L = L_A + (lambda/2) sum F_i (theta_i - theta*_A,i)^2`; SI path integral; MAS
  `Omega_i = mean ||dL/dtheta_i||`. MAS 52.69 average accuracy vs fine-tune 32.67 on 8 tasks.
- Loss of plasticity is measured: ImageNet accuracy 89% -> 77% by the 2000th task
  (arXiv:2306.13812, 2023).
- The stability gap survives even with a full replay buffer; 1 example per class already beats
  every regulariser (arXiv:1904.07734, 2019).
- Hopfield-specific: clipping `|w| <= W` removes catastrophic forgetting but lowers capacity and
  produces a palimpsest (recency-biased) memory (arXiv:2603.09384, 2026).
- Recommended stack for us: clip + column normalisation, then local importance `eta/(1 + lam*omega)`,
  then compressed replay at 1:1 during idle time.

## Finding 4 · HTM and SDR capacity (agent: HTM theory)

- SDR collision probability `1/C(n,k)`: n=1024,k=24 gives ~1 in 10^17 for a single pair.
- Union bound for merged inputs: `E|X| = (1 - (1 - q)^M) n`; at q=0.0234, M=6 keeps sparsity
  near 13%, M=10 pushes it to 21%.
- HTM TM: 2048 columns x 32 cells, ~2% sparsity, ~40 synapses per segment at threshold 15 ->
  ~320k learnable transitions; one pattern survives ~500 temporal contexts.
- Boosting is *not* required (it reduces to a DeSieno conscience term relevant only during early
  SP training).
- Documented HTM weakness: sensitive to temporal noise (one wrong element destroys the high-order
  context) — which is exactly our 0.1% continuation rate.
- NAB score 65.3 vs 47.1 (Twitter ADVec), 35.7 (Skyline), 11.0 (random), and the score is computed
  against a *rolling Gaussian null*, not raw error.

## Finding 5 · Retrieval LMs (agent: kNN-LM / RETRO)

- kNN-LM: `p = lambda p_kNN + (1-lambda) p_LM`, k=32 is enough (k=256 no better), 103M keys ->
  16.12 ppl vs 18.65 without a store.
- Exact-match retrieval is weak: a 4-gram trie gets ~31% top-1 at 5B tokens; interpolated n-gram
  adds only 0.2 ppl.
- **A Kneser-Ney 5-gram gets 39.8% recall@3 on WikiText-103 vs 53.5% for a QRNN; our 8-10% top-1
  is below a 4-gram.** That is the sharpest statement of where we stand: local statistics are
  under-exploited, not the model size.
- Budget design: 64-bit code + 2-byte token = 10 B/entry -> 4M entries in 40 MB, bucketed by the
  top 16 bits, 1-3 us per query on one core.

## Finding 6 · Backprop-free learning at scale (agent: local rules)

| rule | task | result |
|---|---|---|
| Equilibrium propagation (3-phase) | CIFAR-10 | 11.68% error vs BP 11.08% |
| Forward-Forward | MNIST / CIFAR-10 | 1.37% / ~3.3% |
| e-prop | TIMIT | 62.9% vs BPTT 67.1% |
| Target propagation | PTB char-LM | 192-226 vs BPTT 182-188 (worse) |
| Assembly calculus | 2-class / MNIST | 100% / 96% |

No published local-rule system does useful open-vocabulary language modelling at any scale. The
realistic target for us is grammar/FSM-scale sequence learning, not an LM.

## What we are doing wrong (ranked)

1. **Positive-only accumulation.** Every rule that lacks a `-w` or `-k k^T` term converges to a
   constant matrix. Fix: mean subtraction or gated delta (Finding 1, 2).
2. **Read-out is a sum, not a competition.** Modern Hopfield and DAM both apply the non-linearity
   across candidates. Fix: k-WTA or softmax over candidate scores (Finding 1).
3. **No forgetting at all.** The per-batch 0.99999 decay is four orders of magnitude too weak at
   86M tokens. Fix: clip + column normalisation, then local importance weighting (Finding 3).
4. **We compared ourselves to the wrong thing.** A 5-gram beats our next-word accuracy. Fix: match
   the n-gram baseline first, then add retrieval (Finding 5).
5. **No null model in our own benchmarks.** NAB scores against a rolling Gaussian; we report raw
   accuracy. Fix: every benchmark gets a shuffled or rolling-mean control (already true for most).

## Order of work

| # | change | expected effect | cost |
|---|---|---|---|
| 1 | read-out: subtract the mean candidate score, keep top-k (k-WTA) | stops the constant-matrix failure | 1 day |
| 2 | learning: `w += xi*(x_i - mean_x)*(y_j - mean_y) - lambda*w` | bounded weights, fixed point below 1 | 1 day |
| 3 | clip + column renormalisation every N tokens | prevents palimpsest collapse | half day |
| 4 | 64-bit context cache with LRU (4M entries, 40 MB) | exact recent statistics, no saturation | 2 days |
| 5 | match a 5-gram on next-word top-1, then add the cache | the honest baseline we are missing | 2 days |
| 6 | rolling-Gaussian null for the anomaly benchmark | comparable to NAB | half day |

## Sources

Krotov & Hopfield 2016 (1606.01164); Ramsauer et al. 2020 (2008.02217); Demircigil et al. 2017
(1702.01929); Lucibello & Mezerd 2023 (2304.14964); Sacramento & Wichert 2012 (1207.7196);
Schlag et al. 2021 (2102.11174); Yang et al. 2024 (2407.04077); Yang et al. 2023 (2312.06635);
Gu & Dao 2023 (2312.00752); Dao & Gu 2024 (2405.04517); Kirkpatrick et al. 2017 (1612.00796);
Zenke et al. 2017 (1703.04200); Aljundi et al. 2018 (1711.09601); Dohare et al. 2023 (2306.13812);
Mirzadeh et al. 2020 (1904.07734); Chaudhry et al. 2019 (1706.08840); Ahmad & Hawkins 2015
(1503.07469); Ahmad & Purdy 2016 (1607.02480); Cui et al. 2017 (1601.06116); Khandelwal et al.
2020 (1911.00172); Borgeaud et al. 2022 (2112.04426); Wu et al. 2022 (2203.08913); Zhong et al.
2022 (2205.12674); Xue et al. 2021 (2105.13626); Clark et al. 2022 (2111.12681); Yu et al. 2023
(2305.07185); Wang et al. 2024 (2401.13660); Laborieux et al. 2021 (2101.05536); Hinton 2022
(2212.13345); Bellec et al. 2020 (1901.09049); Millidge et al. 2022 (2006.04182); Papadimitriou
et al. 2020 (2102.06560).