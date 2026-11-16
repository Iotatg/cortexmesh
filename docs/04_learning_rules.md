# 04 · Learning rules (implementable specs)

All rules are local: a synapse sees `pre_trace`, `post_trace`, `eligibility`, and a
broadcast scalar. No rule reads another layer's error.

## Notation

- `x_i(t)` — binary activity of input/synapse i at tick t.
- `y_j(t)` — binary activity of cell/column j.
- `w_ij` — synaptic weight (we keep them as integers or floats in [0,1]; for the first
  implementation: `uint8_t` in 0..255, to keep memory tiny).
- `pre_trace_i`, `post_trace_j` — exponentially decaying traces.
- `elig_ij` — eligibility trace (per synapse).
- `m(t)` — modulator scalar (RPE, novelty, ACh...).

## 1. Traces

```
pre_trace_i  ← pre_trace_i * decay_pre  + x_i
post_trace_j ← post_trace_j * decay_post + y_j
decay_pre  = exp(-1 / tau_pre),  tau_pre  ≈ 20 ticks
decay_post = exp(-1 / tau_post), tau_post ≈ 20 ticks
```

## 2. STDP eligibility

```
dt = t_pre_last - t_post_last        # positive = pre before post
if dt > 0:  elig += A_plus  * exp(-dt / tau_plus)
else:       elig -= A_minus * exp(-|dt| / tau_minus)
A_plus = 0.05, A_minus = 0.05 * 1.05   (slightly asymmetric → net LTD, stability)
tau_plus = 2 ticks, tau_minus = 4 ticks
elig *= elig_decay                     # ≈ exp(-1/50) per tick
```

## 3. Three-factor weight update

```
delta_w = eta * elig_ij * m(t)
w_ij   += delta_w
```

`m(t)` is one of:
- `m_rpe = rpe` (reward prediction error, clipped to [-1, 1])
- `m_novelty = novelty` (0..1, from CA1 mismatch)
- `m_ach = 1 + ach_gain * uncertainty` (attention/precision)

## 4. Normalization (stability without a global loss)

Per **cell** (not per layer):
```
sum_w = Σ_i w_ij
target = syn_per_cell * 0.5      # average weight 0.5
w_ij *= target / sum_w           # multiplicative scaling, slow (tau ≈ 500 ticks)
```
This is heterosynaptic/homeostatic scaling. It is what makes pure Hebb stable.

Additionally, per **column**, cap total synaptic input: if a column's total weight exceeds
`max_col_weight`, rescale.

## 5. BCM threshold (selectivity)

```
theta_m_j ← theta_m_j + (1/tau_theta) * (y_j^2 - theta_m_j)     # sliding mean of activity
delta_w_ij = eta * x_i * y_j * (y_j - theta_m_j)
```
Used in the spatial pooler to make columns selective rather than merely correlated.

## 6. Homeostatic firing-rate control

```
rate_j ← (1 - 1/tau_rate) * rate_j + y_j
if rate_j < target_low:  boost_j += boost_inc
if rate_j > target_high: boost_j -= boost_dec
boost_j = clamp(boost_j, 1.0, boost_max)
overlap_j *= boost_j
```
Prevents dead columns and runaway winners.

## 7. Synaptic tagging & consolidation

```
tag_ij ← tag_ij * tag_decay + elig_ij * m(t)      # tag decays over ~1000 ticks
if consolidation_event:
    w_ij += eta_slow * tag_ij
    tag_ij = 0
```
Fast learning is noisy and reversible (tags); slow learning writes tags into structure.
This is the computational version of "systems consolidation".

## 8. Reward prediction error

```
v(s,a) ← v(s,a) + alpha_v * (r + gamma * max_a' v(s',a') - v(s,a))   # simple TD(0)
rpe = r - v(s,a)      # or the TD error itself
```
Dopamine = `rpe` (clipped, broadcast). Cortex learns *which* synapses caused reward via
eligibility; BG learns *which action* to pick.

## 9. Structural plasticity (growth/pruning)

For each cell, per segment:
```
age ← age + 1
if age > max_age and mean_activity < min_activity:
    remove segment (or its weakest synapses)
if active_segment_count > target_segments and prediction accuracy low:
    grow new segment onto co-active cells (dendritic competition)
```
Growth is what allows the system to learn structure, not just weights.

## 10. Inhibition / interneurons (simplified)

- **PV (fast)**: divisive normalization within a column — after activation, scale
  column activity so total = `k_active`.
- **SOM**: if a column bursts (surprise), suppress distal learning for that column this tick
  (branch-specific plasticity gate).
- **VIP**: when novelty is high, disinhibit SOM (open plasticity).

## 11. Precision (attention)

```
precision_region = base + ach_gain * uncertainty_region
prediction_error *= precision_region
```
So "attention" is a scalar gain on prediction error, not a weighting matrix.

## 12. What each module learns (summary table)

| Module | Objective | Rule |
|---|---|---|
| Encoder | (fixed, not learned) | — |
| Spatial pooler | sparse, stable, selective representation | Hebb + BCM + homeostasis |
| Temporal memory | predict next column activation | STDP eligibility + segment growth |
| Hierarchy | minimize layer-wise prediction error | same, plus feedback bias |
| Hippocampus | bind arbitrary conjunctions fast | one-shot Hebb (high η) + pattern separation |
| Basal ganglia | maximize reward | RPE-modulated Hebb (policy) + TD value |
| Cerebellum | minimize supervised error | delta rule (climbing-fiber error) |
| Consolidation | transfer fast → slow | replay × slow Hebb |

## 13. Hyperparameters (phase-1 defaults)

```
n_columns            = 2048
cells_per_column     = 32
sparsity (k)         = 40  (2 %)
syn_per_column       = 128
connected_permille   = 500 (50 % of input space potentially connected)
eta                  = 0.05
eta_fast             = 0.20
eta_slow             = 0.01
tau_pre/post         = 20
tau_elig             = 50
tau_tag              = 1000
boost_max            = 2.0
theta_tau            = 500
inhibition_radius    = 1 column (neighbourhood)
```

These are starting points, to be tuned against the benchmarks in `11_benchmarks.md`.