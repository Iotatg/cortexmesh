# 03 · Architecture: `CortexMesh`

One-sentence version: **a hierarchy of sparse predictive regions, each predicting its next
input SDR, coordinated by a hippocampal fast-learning loop, gated by a basal-ganglia
selector, corrected by a cerebellum-style supervised model, and consolidated during offline
replay.** No attention, no BPTT, no dense layers.

```
                +--------------------------------------------------+
                |                Modulators (broadcast)            |
                |   rpe      novelty/uncertainty      ach/gain      |
                +--------------------------+-----------------------+
                                           |  scalars only
   input stream (tick)                     v
   +---------------->  Gate (thalamus)  ->  Region L1  ->  Region L2  -> ...
   |                                          ^ pred        ^ pred
   |                                          |             |
   |                                   Region L0 (sensory) ---+
   |                                          |
   +---- Motor / action SDR <---- Basal Ganglia (WTA + RPE)
                                    ^
   Hippocampus (fast encoder, pattern sep, replay) --consolidate--> Cortex (slow)
```

## 1. Tick and phases

Discrete time. One tick = one gamma cycle (~10 ms biological). Within a tick:

1. **Sense**: read input, encode to SDR (`Encode`).
2. **Predict**: each region computes predictions from distal segments (`predict()`).
3. **Activate**: k-WTA over columns, biased by predictions (`activate()`).
4. **Learn**: update synapses with eligibility × modulator (`learn()`).
5. **Act**: motor region picks action via BG gate; action effect appended to next input.
6. **Bookkeeping**: decay traces, update homeostatic state, occasionally consolidate.

This ordering is deliberate: prediction must be computed *before* activation so it can bias
which cells win (predictive routing), and learning uses post-activation activity.

## 2. Core object: `Region`

A region is a cortical area. It has:

- `N` columns (each ~`cells_per_column` cells, default 32).
- Proximal synapses (feedforward from an input region) with `syn_per_col` synapses.
- Distal segments (lateral predictions) grown onto active cells.
- Per-column inhibition radius and boosting (homeostatic).
- A stable output SDR (temporally pooled) and a fast output SDR (this tick).
- Two learning rates: `eta_fast` (hippocampal-like, used in `learn_fast`) and `eta_slow`
  (cortical, used in `consolidate`).

```cpp
struct Region {
  uint32_t n_columns, cells_per_column, column_dim;
  SDR (*active)(), (*predictive)(), (*output)();
  void observe(const SDR& input);   // sense + predict + activate
  void learn(const Modulators&);    // local Hebbian/STDP
  void consolidate(size_t iters);   // offline replay-driven
};
```

## 3. Encoding: `Encoder`

Every input is turned into an SDR. Encoders are compositional, not learned end-to-end:

- `CategoryEncoder`: symbol → random sparse code; similar symbols get overlapping codes by
  construction (e.g., hash of symbol into `n` buckets, then top-k).
- `ScalarEncoder`: number → overlapping receptive fields (like place fields).
- `GridEncoder`: (x, y, θ) → concatenation of grid modules (multi-scale periodic).
- `TemporalEncoder`: (symbol, phase, history-hash) → code; order is in the phase bits.
- `SequenceEncoder`: previous k symbols hashed → context bits (this is our n-gram, but
  sparse and composable instead of dense).

Property we require of all encoders: **overlap ∝ similarity**, and **union/intersection are
meaningful** (⊕ = conjunction, ∩ = shared features).

## 4. Prediction and sequence learning: `TemporalMemory`

Per column, `cells_per_column` cells. Each cell has distal segments on other cells
(including its own column) with `delay` ticks. A cell becomes **predictive** if any segment
is active. Then:

- If a predictive cell wins the column → **predicted activation** (correct prediction).
- If a non-predictive cell wins → **burst** (surprise) → all cells in column active.
- Growth: for the winning cell, grow a segment to the currently active cells (with delay 1);
  if no matching segment, grow a new one.
- Adapt: for each active segment on the winning cell, `w += eta * (active - w)` normalized.
- Punish: segments that predicted but did not fire get weakened slightly.

This yields: sequence learning, prediction, anomaly detection (`burst` = anomaly),
variable-order behaviour, and "first-time" surprise.

## 5. Spatial pooling: `SpatialPooler`

Maps an input SDR to a column SDR.

- Each column owns `syn_per_col` synapses to random input bits (connectivity ~`connected_p`).
- `overlap_c = Σ w_i · input_i` (binary weights).
- Inhibition: each column competes with its neighbourhood; top-`k` (or all above
  `min_overlap`) win.
- Learning: winning columns `w += eta` on active inputs, `w -= eta` on inactive; then
  normalize total weight per column (homeostasis).
- Boosting: columns that rarely win get a boost that decays when they win, so all columns
  get used (prevents dead columns).
- BCM threshold `theta_m` per column tracks mean overlap → selectivity emerges.

## 6. Hierarchy: `Cortex`

Regions are wired in a DAG: `L1 → L2 → L3`, with feedback `L3 → L2 → L1` (apical bus).
Each region predicts its *input* region's next output, so learning is unsupervised at every
level (predictive coding). Feedback is a bias, not a drive.

## 7. Hippocampus: fast binding + replay

- `Dentate`: high-sparsity expansion of the cortical state (pattern separation).
- `CA3`: recurrent attractor over the dentate code; completes partial cues.
- `CA1`: compares CA3 output to entorhinal input; mismatch → novelty signal.
- `Replay`: during offline ticks, replay buffered (state, action, reward) tuples in
  compressed bursts; each replay drives *cortical* slow plasticity (consolidation).

## 8. Basal ganglia: action selection

- Each candidate action has an SDR (policy over options).
- Striatum computes `value(action)` from cortical state (learned by RPE-modulated Hebb).
- WTA: pick `argmax` above a softmax temperature; D2/indirect pathway raises the threshold
  when uncertainty is high.
- RPE updates `value` estimates: `rpe = r - v(s,a)`, `v += alpha * rpe`.

## 9. Cerebellum: supervised error correction

- Input = context (cortical state) + motor command.
- Granule expansion → sparse basis.
- Purkinje output = weighted sum (learned).
- Error = target - output (from climbing fiber).
- `w += eta * error * granule_activity` (delta rule), with weight decay.
- Also learns a forward model: predict next sensory state from motor command.

## 10. Working memory

PFC-like region with strong recurrence and slow decay; a gated attractor. DA signal
controls gating (update vs. maintain). Capacity measured in "items" = number of stable
attractor states.

## 11. What is deliberately absent

- No attention mechanism (no Q/K/V, no softmax over sequence).
- No backpropagation, no optimizer state (Adam etc.), no learning-rate schedule.
- No embeddings as dense vectors; symbols are sparse codes.
- No positional encoding as a learned vector; order is phase + delay.
- No batch training, no epochs.
- No global loss function. Objectives are: minimize prediction error (cortex),
  maximize reward (BG), minimize supervised error (cerebellum).