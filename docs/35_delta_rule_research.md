# Bounded-memory sequence learning: delta-rule family vs positive Hebb

## 1. DeltaNet (Schlag 2021; Yang 2024)

Online least squares, `L_t(S) = 1/2 ||S k_t - v_t||^2`:

```
S_t = S_{t-1} - b_t (S_{t-1} k_t - v_t) k_t^T = S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T
```

`(I - b k k^T)` is a Householder projection: it deletes the old value only along `k`, leaving
orthogonal directions untouched, so nothing is driven to a ceiling. The step is proportional to
the error `(v - v_old)`: weights move only while wrong. A positive-only rule lacks this error
term, so every touched weight converges to one point.

State is `d_k x d_v` per head (yours 24x1024x1024); capacity is ~`d_dot` orthogonal keys
(Schlag 4.1).

WikiText-103, 90M: sum rule 31.1/33.0 valid/test ppl, DeltaNet 29.7/31.5. Untruncated
context: sum >260 ppl vs DeltaNet 29.4 test; 0.13M state vs Transformer-XL 6.29M (25.5).
Yang 2024, 1.3B/100B tokens: DeltaNet 16.87/12.21 vs Mamba 17.06/13.89, GLA 17.22/14.47; state
128x vs Mamba 64x, RetNet 512x.

## 2. Gated / decaying updates

RetNet: `S_n = g S_{n-1} + K_n^T V_n`, `g in (0,1)` per head.
GLA: `S_t = G_t ⊙ S_{t-1} + k_t^T v_t`, `G_t in (0,1)^{d_k x d_v}`.
Mamba: `h_t = A_bar h_{t-1} + B_bar x_t`, `A_bar = exp(D A)`, `D = softplus(Wx)`.
Mamba-2/SSD: `A_t = a_t I`, `a_t in [0,1]`; 2-8x faster, 8x larger state.
Gated DeltaNet: `S_t = S_{t-1} (a_t (I - b_t k_t k_t^T)) + b_t v_t k_t^T`.

Decay bounds magnitude: `||S_t|| <~ b ||v|| / (1 - a)`; the price is biased recall,
`S* k = b v/(1 - a + a b) < v`.

## 3. Principle and fixed points

Your `S[i][j] += x_i (1 - S[i][j])` is gradient descent on `1/2 (S-1)^2`, applied only when
`x_i = 1`: every touched weight has fixed point 1 regardless of `j`. After 8M touches each row
has been hit ~190k times, hence 96.9% > 0.9, and `S q` becomes a constant sum, so predictions
collapse. Replacing `(1 - w)` by a forget term bounds everything:

| rule | update | fixed point |
|---|---|---|
| EMA | `S <- a S + h (v - S)` | `h v/(1 - a + h)`; 0 if inactive |
| delta | `S <- S(I - b k k^T) + b v k^T` | `S* k = v` if `a=1`, else `b v/(1-a+a b)` |

`b` must exceed the per-key write share; `a` sets horizon `tau ~ 1/(1-a)` and bounds the norm.
Recall needs `Omega(min(|A|,|B|))` state bits (JRT): 25M entries cannot recall unbounded context.

## 4. Saturation evidence

- Schlag: with capacity 64, linear attention accumulates errors from ~60 associations.
- Gated DeltaNet S-NIAH, 4K: DeltaNet 18.6 / Mamba2 56.2 / GDN 92.2; 8K: 14.4 / 17.0 / 29.6.
  Pure delta cannot erase (collisions); pure decay forgets the needle.
- Repeat After Me: `mem(S) < L log D - 1` gives copy error `> 1/2`; Mamba/LSTM accuracy hits 0
  past training length; transformers need ~100x fewer samples.
- Illusion of State: S4/Mamba are in `TC^0`; A5 needs depth growing with length, a 1-layer RNN
  suffices (no numeric accuracies given).

## 5. Two C++ rules

`k in {0,1}^1024` (24 ones), blocks `b = 0..23`, `S_b` 1024x1024, value `v` 1024-bit. Touch only
rows in `supp(k)`; int8, `a = 255/256` (25M entries = 25 MB; fp32 = 100 MB).

R1, gated delta (best for a specific recent item):

```
for b in active blocks:
  for i in supp(k_b):
    u_i = dot(S_b[i,:], k_b)
    S_b[i,:] = a*(S_b[i,:] - b*u_i*k_b) + b*v_b[i]*k_b
```

Fixed point along `k` is `v`, gain `b/(1-a+a b)`; `a=0.997` (tau~333), `b=0.4` gives 0.998.
Orthogonal content survives; nothing saturates.

R2, decayed Hebb (best for aggregate statistics):

```
for i in supp(k):
  for j in 0..1023:
    S[i][j] = a*S[i][j] + h*(v[j] - S[i][j])
```

Fixed point `h v/(1-a+h)`, bounded; the newest item is attenuated by `b/(lambda+b)`, so this is a
smoothed statistic, not a lookup.

Ranking: R1 first (error-correcting, bounded, exact recent recall); R2 if aggregate statistics
dominate; an Oja rule `S += h (y k^T - y y^T S)` suits statistics-only use. `a < 1` is mandatory:
`1-w` saturation becomes `a w` decay, and the delta term supplies selectivity. Add a ring buffer
of the last ~200k keys (25 MB) for exact recent recall; R1/R2 carry the statistics.

## Citations

- Linear Transformers Are Secretly Fast Weight Programmers. arXiv:2102.11174, 2021.
- Parallelizing Linear Transformers with the Delta Rule over Sequence Length. arXiv:2406.06484, 2024.
- Gated Linear Attention Transformers with Hardware-Efficient Training. arXiv:2312.06635, 2023.
- Retentive Network. arXiv:2307.08621, 2023.
- Mamba. arXiv:2312.00752, 2023.
- Transformers are SSMs (Mamba-2). arXiv:2405.21060, 2024.
- Gated Delta Networks. arXiv:2412.06464, 2024.
- The Illusion of State in State-Space Models. arXiv:2404.08819, 2024.
- Repeat After Me. arXiv:2402.01032, 2024.
- Just read twice. arXiv:2407.05483, 2024.