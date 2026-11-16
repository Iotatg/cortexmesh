# 13 · Discoveries from our own experiments

Everything here was measured with `./bin/experiments <name>` (source: `tools/experiments.cpp`).
Numbers are from the current code, single run, deterministic seeds. These are findings about
*our* implementation, not claims about biology — but each one changed a design decision.

## D1 · Capacity law: `columns ≥ 32 × items`

Setup: input 1024, k=20, columns = items × {8,16,32,64}, 20 active columns, 64 synapses,
16 cells/column, 25 epochs, accuracy = fraction of transitions that do not burst.

| items | 8× | 16× | 32× | 64× |
|---|---|---|---|---|
| 4 | 0.750 | **1.000** | 1.000 | 1.000 |
| 8 | 0.375 | **1.000** | 1.000 | 1.000 |
| 16 | 0.375 | 0.625 | **1.000** | 1.000 |
| 32 | — | 0.719 | **1.000** | 1.000 |
| 64 | — | — | 0.656 → 0.781 (16×) | **1.000** (64×) |

Clean reading: at **8× columns per item** accuracy collapses to 0.375–0.75; at **32×** it is
1.000 for every size tested. The usable capacity is therefore

```
columns_needed ≈ 32 × items        (at 20/1024 active, 64 synapses, 16 cells)
```

Practical rule: size the region from the number of distinct states you expect, not from a
fixed 2048/4096 default. This replaces the earlier "columns ≥ 40 × items" guess.

## D2 · Boosting causes representation drift (and we removed it)

Setup: 8 items, 1024 columns, 20 active, 64 synapses, 15 epochs; drift = overlap of the
code at epoch 0 vs after training, averaged.

| boost_max | boost_inc | drift (1024 cols) | drift (4096 cols) |
|---|---|---|---|
| 1.5 | 0.02 | 0.737 | 0.538 |
| 1.0 | 0.00 | **0.988** | — |
| 1.0 | 0.02 | **0.988** | — |

Boosting is what keeps silent columns alive in random-input tests, but it **reshuffles the
winners on every tick**, so the region never settles into a stable code. With boosting off,
codes are stable (0.988) and sequence learning works.

Decision: `boost_max = 1.0`, `boost_inc = 0.0` by default. Homeostasis is handled by fixed
synapse count + weight normalization, not by winner boosting. If a column is starved in a
deployment, the fix is capacity or a one-time re-seed, not continuous boosting.

## D3 · Long eligibility traces destabilise sparse value learning

Setup: `Valuation` over 2048-bit SDRs, reward after a delay of 1/5/20 ticks, 300 reps,
score = `V(A) − V(B)` (ideal 1.0).

| trace_decay | delay 1 | delay 5 | delay 20 |
|---|---|---|---|
| 0.00 (no trace) | **0.962** | 0.895 | **0.709** |
| 0.50 | 0.927 | 0.811 | 0.551 |
| 0.70 | **3.478** | 2.140 | 0.731 |
| 0.85 | 1.756 | 1.076 | 0.366 |
| 0.95 | 0.604 | 0.365 | 0.122 |
| 0.99 | 0.143 | 0.080 | 0.025 |

Interpretation: a long trace keeps *old* active bits eligible, so rewards for A bleed into B
and vice versa. Worse, the readout can overshoot (`3.478` is impossible for a value in [0,1]).
So eligibility must be **short and gated by the modulator**, not a long memory. This is the
opposite of the usual "eligibility traces solve delayed credit assignment" intuition in our
setting — because our value head is a sparse linear readout with no nonlinearity.

Decision: default `trace_decay = 0.85` is too long; use `0.0–0.5` for the value head, and put
the delay handling in the **replay/hippocampal** system instead (D4).

## D4 · Forward replay consolidates; reverse/random replay do not

Setup: learn A (8 items), measure, learn B (8 items), then replay A in forward/reverse/random
order (4 passes) or not at all, measure A again.

| replay order | accuracy before B | accuracy after |
|---|---|---|
| forward | 1.000 | **1.000** |
| reverse | 1.000 | 0.875 |
| random | 1.000 | 0.875 |
| none | 1.000 | 0.875 |

Only **forward** replay restores A. Reverse/random replay is no better than nothing, and can
be worse (it re-teaches wrong transitions). This matches the biological finding that
sharp-wave-ripple replay preserves sequence order and that reverse replay serves a different
function (reward/credit assignment).

Decision: replay buffer stores ordered sequences; consolidation replays them forward. Do not
shuffle replay data.

## D5 · Burst count is a calibrated anomaly/uncertainty signal

Setup: learn 16 items, then corrupt inputs by flipping `f` random bits; anomaly = 1.0 when a
tick bursts.

| bit flips | mean anomaly | burst rate |
|---|---|---|
| 0 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 |
| 5 | 0.062 | 0.062 |
| 10 | 0.125 | 0.125 |
| 20 | 0.312 | 0.312 |
| 40 | 0.750 | 0.750 |
| 80 | 1.000 | 1.000 |

Monotone and usable directly as the `novelty` input to the modulator system. It is also a
cheap uncertainty estimator: no extra network, no sampling.

## D6 · Conjunction value heads interfere; per-action heads do not

Setup: two states, two actions; train `Q(sA,aX)=1`, `Q(sB,aY)=0.2`; measure the gap
`Q(sA,aX) − Q(sA,aY)` for the *untrained* action (ideal 0.8, 0 = full interference).

| value head | code_size/k | gap |
|---|---|---|
| conjunction `(state,action)` | 2048/8 | 0.891 |
| conjunction | 8192/16 | 0.395 |
| conjunction | 32768/16 | 0.395 |
| **per-action head over state** | any | **1.000** |

Reason: `(sA,aX)` and `(sA,aY)` share the state half of the code, so the linear readout cannot
separate them; the untrained action inherits ~half the trained value. Splitting the value into
**one head per action** (striatum-like channels) removes the interference completely.

Decision: `DualController` now keeps `action_values_[action.hash()]` and evaluates
`predict(state)` per action. Conjunction is still used for binding (hippocampus), not for
scalar readouts.

## D7 · Focus follows an inverted-U (measured)

Setup: 12 items, 2048 columns, 4 epochs, `Region::set_focus(g)`; `g` scales the temporal-memory
learning gain and the spatial-pooler tie jitter (`0.02 × g`).

| focus g | accuracy |
|---|---|
| 0.2 | 0.750 |
| 0.5 | 0.750 |
| 1.0 | 0.750 |
| 2.0 | 0.750 |
| 4.0 | 0.667 |
| 8.0 | 0.583 |
| 16.0 | 0.417 |

Interpretation: more gain is not better. Beyond `g≈2`, the added tie jitter makes the spatial
code unstable, and prediction collapses. This is the computational version of the LC-NE
inverted-U (Aston-Jones & Cohen): moderate arousal sharpens, extreme arousal scatters.

Decision: operating point `g ≈ 1`, and `set_focus` should be used as a *regulator* (raise when
uncertain, lower when confident), not as a "more is better" knob.

## D8 · What this buys us (design rules that came out of the numbers)

1. **Size regions from state count**: `columns ≥ 32 × items`.
2. **No winner boosting**: it destroys code stability; use fixed connectivity + normalization.
3. **Short eligibility, gated by RPE**: delay is handled by replay, not by long traces.
4. **Forward-only replay** for consolidation; never shuffle replay.
5. **Burst rate = uncertainty**: free anomaly signal, feeds salience/attention.
6. **Per-action value heads**, not conjunction readouts.
7. **Focus is a regulator** with an optimum near 1.0, not a monotonic knob.

## D9 · Experiment plan (completed, results below)

## D9 · X1–X5 results (all five experiments run)

### X1 · Replay scaling

15 epochs A, 15 epochs B, then measure A. `columns = 32 × items`.

| items | none | replay 5 epochs | replay 10 epochs | interleaved (1 A step per 4 B) |
|---|---|---|---|---|
| 8 | 0.375 | 0.875 | **1.000** | 0.250 |
| 16 | 0.312 | 0.938 | **1.000** | 0.250 |
| 32 | 0.156 | 0.844 | **0.969** | 0.281 |

Findings:
1. Replay scales to 32 items; more replay epochs is better (`end10` ≈ 1.0).
2. **Interleaved replay is harmful** (0.25–0.28): A and B compete for the same columns
   during training, so the code for A never stabilises. Replay must be a **separate
   consolidation phase after** the new episode, not mixed into it.
3. Without replay the gap grows with item count (0.375 → 0.156), i.e. interference is
   the dominant failure, not capacity.

Decision: replay = post-episode phase, forward order, 5–10 epochs.

### X2 · Interoception as a value channel

Reward is determined by body state only (`calm → 1`, `stressed → 0`); 6 states × 2 body
conditions, 400 trials, linear value head.

| input to value head | MAE |
|---|---|
| state SDR only | **0.500** |
| state SDR + interoceptive bits | **0.032** |

Findings:
1. Interoception adds a genuinely predictive channel: MAE drops 15×.
2. A linear head **cannot learn XOR-style conjunctions**. A first attempt used
   `reward = (state parity == calm)`; both models stayed at 0.500 because the reward is a
   product of two features. Conjunction features (or a second-order region) are required
   for state × body rules.

Decision: interoception is wired into the value head as extra input bits; conjunctive
valuation rules need a region, not a bigger linear head.

### X3 · Hierarchy

L1 (1024 cols) learns 16 items; L2 (512 cols) learns the *sequence of L1 column codes*.
20 epochs, then measure burst rate.

| level | accuracy (no burst) |
|---|---|
| L1 | 1.000 |
| L2 | 1.000 |

Findings: L2 learns a predictive model of L1's output. This is the substrate for feedback
(apical predictions that gate L1), but feedback is **not implemented yet** — no burst
reduction below 5 % has been demonstrated, because there is no path from L2 to L1.

### X4 · Memory ceiling

`epochs = 15`, accuracy after training.

| items | 32× columns | 64× columns |
|---|---|---|
| 64 | 0.953 | 0.969 |
| 128 | 0.930 | 0.969 |
| 256 | **0.812** | **0.988** |

Findings: there is no hard "memory ceiling" at 32×; the law is simply that larger item sets
need **64× columns**, not 32×. 32× is enough up to ~64 items, then accuracy decays
(0.953 → 0.930 → 0.812). 64× stays ≥ 0.969.

Decision: capacity rule becomes `columns ≥ 32 × items` for `items ≤ 64`, `≥ 64 × items`
above that (with 20/1024 active, 64 synapses, 16 cells/column).

### X5 · Pruning

Train 25 epochs, `prune(2.0)`, then 10 epochs of relearning. `removed` = synapses dropped.

| items | columns | no prune | prune | removed |
|---|---|---|---|---|
| 8 | 64 (8×) | 0.375 | 0.375 | 769 |
| 8 | 128 (16×) | 0.500 | **1.000** | 934 |
| 16 | 128 (8×) | 0.500 | **1.000** | 1578 |
| 16 | 256 (16×) | 0.812 | 0.938 | 2009 |

Findings:
1. Pruning weak synapses is **safe** if the removed synapses are already below the
   permanence threshold: prediction is unchanged immediately after prune.
2. Pruning **plus relearning** recovers accuracy at low column counts (0.5 → 1.0), because
   freed capacity lets the surviving columns specialise.
3. Two bugs found and fixed while running X5:
   - `homeostat` targeted the *current* synapse count, so pruning rescaled the remaining
     weights down and killed columns. Target is now the **nominal** size
     (`synapses_per_column × permille`).
   - Pruning all weak synapses left some columns empty; prune now keeps a floor of
     `max(8, nominal/2)` strongest synapses per column.

Decision: `SpatialPooler::prune(min_perm)` keeps a connectivity floor; use it as an
offline consolidation step, then relearn briefly.

### D10 · Overtraining drift (new failure mode)

After convergence, extra epochs **reduce** accuracy: 16 items / 512 columns, 25 epochs →
1.000; +10 epochs → 0.938; prune +10 epochs → 0.875.

Cause: permanent weights saturate and columns drift toward each other (no global loss to
stop it). Decision: train to convergence, not past it; add an early-stop on burst rate
when a plateau is reached.

## D11 · Apical feedback (X6, negative result)

Setup: L1 (1024 cols) learns 16 items; L2 (512 cols) learns the sequence of L1 column codes.
At test, L2 predicts the *current* item from the previous clean L1 code and feeds it back to
L1 as an apical bias (`SpatialPooler::activate(input, feedback)` with `feedback_gain`).

Gated feedback (applied only when the previous step burst), `noburst` = fraction of steps
with no burst:

| gain | 0 flips | 5 flips | 10 flips | 20 flips |
|---|---|---|---|---|
| 0.0 | 1.000 | 0.812 | 0.688 | 0.500 |
| 0.5 | 1.000 | 0.812 | 0.688 | 0.438 |
| 1.0 | 1.000 | 0.812 | 0.625 | 0.438 |
| 2.0 | 1.000 | 0.688 | 0.688 | 0.312 |
| 4.0 | 1.000 | 0.375 | 0.688 | 0.062 |

Recovery (overlap with the clean column code) stays ≈ 0.96–1.00 at every gain.

Findings:
1. Feedback **never improves** robustness here; gain ≥ 2 actively hurts.
2. The SP is already robust to 20 bit flips (recovery 0.96), so the apical bias only adds
   noise: the prediction is a *code*, not a per-synapse prior, and it can pull the wrong
   columns when L2 is imperfect.
3. Correct alignment was verified (L2 predicts the current item from the previous one), so
   this is not a one-step misalignment bug.

Decision: keep the feedback API (it is the substrate for hierarchy), but do **not** use a
fixed apical gain. Feedback must be gated by *uncertainty* and probably applied at the
synapse level (per-synapse apical depolarization), not as a column-level multiplier.

## D12 · Variable-order / anomaly (X7)

Ambiguous list: `x → a`, `x → b`. Context list: `p,x → a`, `q,x → b`. 5 seeds, first pass
after training.

| case | trained successor | novel continuation |
|---|---|---|
| ambiguous (`x,a` / `x,b`) | 0.000 burst | `x,c` → **1.000** |
| context (`p,x,a` / `q,x,b`) | 0.000 burst | `p,x,b` (wrong context) → **0.600** |

Findings:
1. TM builds a *set* of possible successors: a trained successor never bursts, a novel one
   always bursts. This is a free, calibrated anomaly signal.
2. Context (distal 2-step) is used, but only partially (0.6): with `new_segment_min = 3`
   and growth from 8 random prev cells, the context cells are diluted.
3. Intermittent bursts (~20 % in a 5-step probe) appear even on trained transitions. Cause:
   after a successful prediction only the *winning* cell is active, so distal segments grown
   from a bursting previous column (16 active cells) no longer reach threshold. Standard HTM
   has the same winner-only propagation; it needs either predictive cells in the previous
   state or a lower threshold.

Decision: anomaly = burst rate is usable now (AUC next). Context needs either a longer
training phase or a dedicated 2-step segment growth policy.

## D13 · Spatial pooler invariant collapse (found by the simulation test)

Symptom: with a factored state code (x + y + dir + food sub-codes), the SP produced the same
20 columns for different positions, and a 2-cell jump produced **zero** bursts.

```
code overlap 4-6 = 20/40
column overlap 4-6 = 20/20
jump-by-2 burst = 0/11
```

Cause: constant sub-codes (y, dir, food) give a stable overlap to many columns, so the
top-k winners are the invariant columns for every input. The varying part (position) is
dropped.

Fix: encode the full state as a single sparse assembly (no shared sub-codes). Measured after the fix:
column overlap 0/20, jump-by-2 burst 0.63, clean burst 0.003, sensor-noise burst 0.003.

Rule: **never concatenate sub-codes that share a constant component**. Use a conjunction or
a single sparse code for the full state, or give each stream its own region.

## D14 · Two-stream simulation detector (D13 applied)

Pose stream (single sparse assembly) + patch stream, clean vs glitches vs sensor noise:

| condition | pose burst | patch burst |
|---|---|---|
| clean | 0.003 | 0.000 |
| noise 1 bit | 0.003 | 0.000 |
| noise 3 bits | 0.003 | 0.007 |
| teleport | 0.950 | 0.580 |
| speed2 / speed3 | 0.630 / 0.610 | 0.000 |
| spin | 0.110 | 0.000 |
| pass_wall | 0.010 | 0.000 |
| food_dup | 0.000 | 0.000 |

Findings:
1. Pose stream detects kinematic violations with ~0 false positives on sensor noise.
2. Rare events (pass_wall) are diluted by the rate metric; persistence (`run_max`) separates
   teleport (15.8) from noise (1.0).
3. Patch stream is near-constant in open space and misses object events; needs a richer
   encoding.
4. No action input: a forward model `(state, action) -> next state` is the next step.

## D15 · Language without tokens (feature-grounded word codes)

Replaced `Encoder::token(id)` (transformer-style vocabulary) with `WordEncoder`: a word is a
superposition of phone codes (unique part + place/manner/voicing/height), 8 grounded concept bits,
and valence/arousal. Context is a shift-binding of previous word codes; readout is one linear unit
per word trained by `w += lr * (target - score) * ctx` (no softmax/backprop).

Measured (small 11-sentence Hinglish corpus):

| metric | value |
|---|---|
| overlap, words sharing ≥2 phones | 4.39 |
| overlap, unrelated words | 2.66 |
| ambiguous contexts | 3/31 |
| sequence train top-1 / top-3 / MRR | 0.667 / 0.939 / 0.803 |
| sequence test top-1 / top-3 / MRR | 0.250 / 0.500 / 0.368 |
| grounding word → concept | 1.000 |
| grounding concept → word | 0.375 |
| feeling (body → word), 0–8 bit flips | 1.000 |
| generated sentence | `mujhe bhookh lagi hai` |

Findings:
1. Feature codes give real similarity structure (1.65× unrelated); ids give none.
2. Prediction works on seen contexts; test drop is ambiguity, not encoding.
3. Interoception → word is noise-robust (body code can lose 8 bits and still name the feeling).
4. Concept → word is weak because 8 concept bits overlap (`body+need` shared).
5. No motor/audio loop yet: phones are features, not produced speech.

## D16 · Correlated inputs break the SP/TM stack (simulation lesson)

While replacing the simulation's token pose with a grid-cell place code, clean-run burst jumped
from 0.017 to 0.61–0.77 and teleport separation collapsed:

| pose code | clean burst | teleport burst |
|---|---|---|
| random 20-bit token | 0.017 | 0.960 |
| grid 256-bit + dir block | 0.187 | 0.150 |
| grid 32-bit place only | 0.257 | 0.230 |
| sparse grid + dir conjunction | 0.770 | 0.710 |
| grid 256-bit, 8192 cols/128 syn | 0.610 | 0.630 |

Cause: neighbouring grid codes overlap ~50 %, so SP columns overlap; the TM cannot assign a unique
successor. Random/orthogonal codes give unique columns and learn perfectly. The SP/TM stack needs
near-orthogonal inputs (or a much larger pooler + more training). For the simulation the state is
now a hash-based **state assembly** (`state:<id>`), not a language token.

## D17 · Sensorimotor speech loop (analysis-by-synthesis)

`SpeechModel`: articulation code (jaw/lip/tongue/tip/velum/voicing + phone identity) and acoustic
code (F1/F2/F3/pitch + phone identity); two Hebbian memories learn art→sound and sound→art.

| metric | value |
|---|---|
| forward top-1 (23 phones) | 1.000 |
| inverse top-1 | 1.000 |
| forward top-1 with 2/4/8 articulation bit-flips | 1.000 / 1.000 / 1.000 |
| articulation overlap p/b, t/d | 10/32, 10/32 |
| sound overlap ee/ii | 3/32 |

Findings:
1. A closed sensorimotor loop is learnable with local Hebbian rules and no labels.
2. Noise in the motor code does not break perception — the inverse model is robust.
3. Similar phones share bits (p/b, t/d), so the space has phonetic structure.
4. **Theta-gamma phase coding** separates `AB` from `BA` (contexts overlap 0), while plain bundling
   makes them identical (overlap 48/48) and order is lost.

## D18 · Audit: model vs code (controls)

| control | result |
|---|---|
| speech permuted labels (art[i]→sound[i+7]) | 23/23 |
| speech features only (identity removed) | 10/23 |
| word code vs concept overlap | 1.50/24 (no leak) |
| word → shuffled concept | 24/24 |
| held-out "pyaas" concept | hit |
| body vs word overlap | 0.80/512 (no leak) |
| body → shuffled word | 5/5 |

Verdict: learning is real (arbitrary mappings are learnable), but competence is narrow: speech
leans on the 12-bit identity part, language is supervised on a small corpus, and the simulation
memorizes exact `(x,y,dir)` states. No hard-coded answers were found.

## D19 · Word-level production + co-articulation

Word code ↔ audio summary (OR of phone acoustic codes), 25 words.

| metric | value |
|---|---|
| hear top-1 / top-3 | 21/25, 24/25 |
| say top-1 / top-3 | 13/25, 19/25 |
| held-out "pyaas" rank | 3/25 |
| co-articulation: k vs kaa overlap | 18 → 4 |
| context-aware inverse | 86/170 |

Findings:
1. Hearing is much easier than saying; the audio summary is a bag of phones, so words sharing
   phones collide.
2. Held-out words generalize through shared phones.
3. Context changes the acoustic target (co-articulation), but OR-binding of prev+current is weakly
   separable (51%).
4. Order is not in the audio summary; sequence/phase coding is the next step.

## D20 · Ordered audio + syllable layer

| representation | hear top-1 | say top-1 | reversal overlap |
|---|---|---|---|
| bag of phones | 21/25 | 13/25 | high |
| phase-ordered phones | 24/25 | 18/25 | 24.6 |
| syllable layer | 24/25 | 21/25 | — |

Findings:
1. Phase-binding positions makes word order explicit (reversal overlap 24.6).
2. Syllables improve production (13 → 21) by giving the model a middle level.
3. Hearing is near-perfect; saying is still limited by shared phones.
4. Syllabification is hand-written; learned boundaries are next.

## D21 · Learned boundaries and learned auditory categories

`tools/phonotactics.cpp`: within-word phone bigrams -> surprise -> boundary detection;
noisy formants -> k-means (k=23) -> categories. Details in `docs/20_phonotactics.md`.

| item | value |
|---|---|
| boundary precision / recall / F1 | 1.000 / 0.944 / 0.971 |
| vowel-nucleus baseline F1 | 0.022 |
| auditory cluster purity | 0.735 |
| clean-target recognition | 17/23 |

Findings:
1. Word edges are prediction-error peaks: surprise threshold 2.80 gives F1 0.971.
2. The hand vowel rule is worse than chance (0.022); most edges follow a vowel-final word.
3. Acoustic categories are learnable without labels: 73.5% purity, 17/23 phones correct.
4. Remaining confusions are place/manner neighbors (p/b, t/d, s/sh), same set as the
   supervised inverse model, so the categories are acoustic, not memorized.

## D22 · Knowledge induction and the interference knob

`tools/curriculum.cpp`: facts as Hebbian matrices, multi-hop chaining, feature-bundle vs
hash entity codes, replay consolidation. Details in `docs/21_knowledge_induction.md`.

| condition | A hit@1 | B hit@1 | C hit@1 | A multihop overlap |
|---|---|---|---|---|
| hash entities | 1.000 | 1.000 | 1.000 | 1.000 |
| feature-bundle, no replay | 0.889 | 1.000 | 1.000 | 1.000 |
| feature-bundle, replay A 6x | 1.000 | 0.444 | 0.500 | 1.000 |
| shuffled control | 0.000 | — | — | — |

Findings:
1. Multi-hop chaining works with no labels: entity -> isa -> category -> needs -> thing.
2. Shuffled outputs sit at chance (0.000 hit@1, 0.069 overlap).
3. Orthogonal codes retain perfectly and generalise zero; overlapping codes generalise
   and forget. Retention and generalisation are the same knob.
4. Related new facts can *strengthen* old ones (A recovers to 0.889 after C).
5. Block replay protects the replayed set and destroys newer ones (B 0.444, C 0.500).
6. `AssociativeMemory::recall` had a random-init bug (`SDR(out_, k_)`), fixed to
   `SDR(out_, 0)`; it had depressed all code-level recall metrics.

## D23 · Nested phases and co-articulation in the code

`tools/sequence.cpp`: bag vs flat4 vs nested phase order, co-articulated phone codes,
held-out word, slow-phase generalisation. Details in `docs/22_ordered_speech.md`.

| mode | hear top1 | say top1 | reversal overlap | context overlap |
|---|---|---|---|---|
| bag | 18/24 | 15/24 | 0.993 | 0.993 |
| flat4 | 22/24 | 18/24 | 0.326 | 0.993 |
| nested | 21/24 | 18/24 | 0.309 | 0.993 |
| nested + coarticulation | 21/24 | 19/24 | 0.309 | 0.908 |

Findings:
1. Phase rotation encodes order: reversal overlap 0.99 -> 0.31.
2. Nested is not better than flat at 2 syllables; the win should appear at 3+.
3. Co-articulation only survives in nested mode (context overlap 0.908).
4. Held-out "pyaas" ranks 1st of 24 by shared phone codes.
5. Slow half carries syllable identity at 1.87x chance (0.283 vs 0.151), but diluted.

## D24 · Streaming EMA language model on real text

`tools/streamlm.cpp`, corpus `Salesforce/wikitext` (`wikitext-2-raw-v1`, train split,
17,187 sentences, 1,742,067 tokens, 5k vocab). One pass, EMA Hebbian, no stored pairs.
Details in `docs/23_efficient_language_research.md`.

| xi | first-half top1 | second-half top1 | all top1 | top10 | MRR | peak RSS |
|---|---|---|---|---|---|---|
| 0.010 | 0.138 | 0.044 | 0.091 | 0.202 | 0.131 | 15 MB |
| 0.002 | 0.159 | 0.151 | 0.155 | 0.275 | 0.201 | 15 MB |
| 0.001 | 0.163 | 0.156 | 0.160 | 0.289 | 0.211 | 15 MB |

Findings:
1. One pass beats two epochs: 0.160 top-1 vs 0.117 batch.
2. Memory is bounded by the matrix, not the corpus: 15 MB at 1.74M tokens.
3. The batch tool holds every pair: 127 MB at 283k tokens, ~0.8 GB extrapolated.
4. High learning rate causes within-run forgetting (0.138 -> 0.044); xi=0.001 is stable.
5. SNAP-style gating helps at xi=0.01 but does not fix a rate that is too high.
6. Bit-frequency normalization makes no measurable difference (0.201 vs 0.201).

## D25 · Evidence protocol and module bench

`tools/bench.cpp` measures the five questions for SP, TM and associative memory, each
with a control. `tools/streamlm.cpp` adds cross-corpus transfer. Details in
`docs/24_evidence_protocol.md`.

| module | evidence | failure found |
|---|---|---|
| SP | stability 0.994, noise 0.795 at 16 flips, code overlap 0.040, class info 1.000 | 209/512 dead columns at 400 samples |
| TM | next-symbol 0.878 vs 0.009 prev / 0.051 freq | burst anomaly 0.412 clean vs 0.521 corrupt (weak) |
| assoc | recall 1.000 at 100, 0.868 at 400; retention 0.982; composition X=1.000, Y=1.000 | none at this scale (needs 2K-wide probe) |
| LM | inv top1 0.105 same-domain / 0.091 transfer, 15 MB | reported top-1 was UNK-inflated (D27) |

Findings:
1. SP learns and is noise-robust, but column death is a capacity failure at small input sets.
2. TM predicts transitions perfectly but its burst signal is saturated by 60k segments.
3. Associative memory retains well and superposes correctly when the probe is a full 2K bundle (D28).
4. The streaming LM transfers across corpora at 86% of same-domain top-1 (in-vocabulary only).
5. Every "win" has a control; every failure is reported, not hidden.

## D26 · Temporal memory burst needs a sparse, learnable context

Bench originally used an order-2 Markov source: 4096 contexts over 512 columns with 4
cells each (2048 cells) - not learnable at that capacity, so every run looked like
"burst is saturated". Switching to an order-1 source (64 contexts) and fixing the burst
path gave:

| variant | predict | burst clean | burst corrupt | segments |
|---|---|---|---|---|
| all-cells burst, 16 seg/cell, thr 6 | 0.935 | 0.386 | 0.493 | 59k |
| all-cells burst, 8 seg/cell, thr 6 | 0.878 | 0.412 | 0.521 | 30k |
| single-cell burst, 8 seg/cell, thr 6 | 0.974 | 0.453 | 0.894 | 3.5k |

Rules learned:

1. **Capacity law for sequences**: `cells_per_column x max_segments_per_cell >= number of
   distinct contexts`, otherwise eviction churn keeps the region bursting.
2. **Burst context must be rich enough to grow a segment**: a single active cell produces
   one-synapse segments, which can never reach an activation threshold of 2. That is why
   the single-cell variant (best anomaly separation) breaks `Region` tests.
3. **Do not grow a new segment for every winner**: that block churned segments each step
   and lowered accuracy; removed.
4. **Anomaly is now a fraction** of active columns bursting, not a binary flag.
5. Honest verdict: prediction is strong (0.88-0.97 vs 0.009 prev / 0.051 freq), but burst
   is a **weak** anomaly signal with the all-cells context (0.41 vs 0.52). A dedicated
   novelty detector is still needed.

## D27 · Top-1 accuracy was inflated by UNK

On TinyStories, 22.5% of targets are out of vocabulary. `streamlm` ranked over all 3k codes
including UNK, so the model scored 0.226 top-1 mostly by **predicting UNK**. Ranking over
known words only:

| set | oov | inv top-1 | inv top-10 | mrr |
|---|---|---|---|---|
| wikitext tail (same domain) | 0.260 | 0.105 | 0.183 | 0.105 |
| TinyStories (transfer) | 0.225 | 0.091 | 0.151 | 0.093 |

Transfer is 86% of same-domain - the cross-corpus claim survives, but the absolute number
is ~10%, not ~22%. Also tested and rejected:

| variant | transfer top-1 |
|---|---|
| word hash codes | 0.226 (incl. UNK) / 0.091 (known only) |
| character-trigram codes | 0.213 |
| trigram + identity hybrid | 0.213 |
| trigram context (prev1 + rot(prev2)) | 0.105 (no gain over bigram) |

Sub-word codes do not help at 3k vocab: shared trigrams add confusability without adding
coverage. The next lever is a character-level output space, not word-level codes.

## D28 · Superposition works; the probe was wrong

`assoc,composition` reported X=1.000 / Y=0.000. The cause was not the memory: the probe
`A|B` was truncated to K bits (24 of 48), so half of Y was thrown away before recall.
With the full 2K probe and `recall(in, 2K)`:

| probe | overlap X | overlap Y |
|---|---|---|
| 24-bit truncation | 1.000 | 0.000 |
| 48-bit bundle | 1.000 | 1.000 |

`AssociativeMemory::recall(in, k)` added for variable-width readout. A superposition is
2K wide by construction - truncating it to K is a bench bug, not an architecture limit.

## D29 · The LM produces words, not sentences

Generation was added to `streamlm` (`generate`, `seed`, `temp`, `npos` args). Samples:

| model | sample |
|---|---|
| TinyStories, 2k vocab, temp 1.0 | "once sick him mighty walking never to house berries farm ... pony little brave" |
| wikitext, 3k vocab, bigram | "the 1920 germany starred scored 48 columbus passage association" |
| wikitext, 3k vocab, 4-word positional | "the train bridge guides yue 48 columbus dictator authority" |

Findings:

1. Vocabulary and collocations are real (entities, numbers, domain words) - the model has
   lexical knowledge from the corpus.
2. No syntax: output is a bag of likely words. A linear context-to-code readout with EMA
   learns co-occurrence, not grammar.
3. Positional 4-gram context (4x512 bits, 3 MB) gave **no** accuracy gain over bigram
   (inv top-1 0.0725 vs 0.0757) - longer context does not help this readout.
4. Temperature 0.3 collapses to loops ("the singapore the singapore"); 1.0 diversifies.

## D30 · Character-level output removes OOV (but not the syntax problem)

`tools/charlm.cpp` predicts the next character with the same local Hebbian matrix, positional
context (k=2 chars, N=256, K=8, 48-char alphabet).

| set | top-1 | top-5 | MRR | oov |
|---|---|---|---|---|
| train all (1.73M chars) | 0.338 | 0.653 | 0.478 | 0.000 |
| train first half | 0.293 | - | 0.451 | 0.000 |
| train second half | 0.383 | - | 0.506 | 0.000 |
| TinyStories (no retraining) | 0.438 | 0.622 | 0.533 | 0.000 |

Baselines: previous character 0.020, most frequent character 0.197. Peak RSS 11 MB.

Findings:
1. OOV is gone by construction (22.5% word-level -> 0.000), and transfer *improves*
   (0.438 vs 0.105 word-level in-vocabulary).
2. Learning continues across halves (0.293 -> 0.383): no catastrophic forgetting at this size.
3. Longer context hurts: k=2 0.283, k=3 0.250, k=4 0.231 - the readout cannot use more context.
4. Generation is still sub-lexical, not sentences (D29 verdict unchanged).
5. Per-character and per-word accuracy are not directly comparable; the OOV=0 claim is partly
   definitional (0.57% of eval characters fell outside the training alphabet).

## D31 · SDM is a real dense-matrix alternative, with a capacity cost

`include/bm/sdm.hpp` + `tools/sdm.cpp`: hard-location memory, A=4096 addresses, C=1024-bit
registers, threshold 28, top-40 selection, XOR write, load-normalised majority read.

| question | SDM | dense `AssociativeMemory` |
|---|---|---|
| recall >= 0.9 up to | 1500 patterns | 3200 patterns |
| retention (A then B) | 1.000 -> 0.998 | 0.999 -> 0.982 |
| noisy cue 25% flipped | 0.953 | ~1.000 |
| noisy cue 50% flipped | 0.509 | 0.970 |
| RAM at this size | 1.03 MB | 4.19 MB (3.98x) |
| write / read | 126 / 141 us | 27.6 / 50.2 us per pattern |
| correlated codes (control) | 1.000 recall / 0.886 precision | 0.799 / 0.799 |

Controls: uncorrelated 0.031, cross-talk 0.040, shuffled 0.048 (chance 0.039).

Findings:
1. SDM wins on RAM (4x) and on **correlated** codes, where the dense matrix collapses.
2. SDM loses on orthogonal capacity (~0.37 patterns per location) and is ~2.4x slower.
3. The literal Kanerva "ones > selected/2" read rule fails at 10% noise (0.525) and 25%
   (0.000); the normalised rule is what makes it work. Both are documented.
4. As an attention substitute: SDM top-1 0.98 / value 0.696 vs dense 1.00 / 0.863.

## D32 · TM is a weak per-step novelty detector (capacity-bound)

`tools/novelty.cpp` evaluates four signals on a learnable order-1 source with injected
anomalies (random, skip, stutter), 12,000 steps, 10% injection.

| signal | random | skip | stutter | pooled |
|---|---|---|---|---|
| burst fraction | 0.753 | 0.687 | 0.567 | 0.666 |
| prediction error | 0.753 | 0.687 | 0.567 | 0.666 |
| unexpectedness | 0.753 | 0.687 | 0.567 | 0.666 |
| surprise EWMA (a=0.2) | 0.817 | 0.745 | 0.572 | 0.711 |
| shuffled labels (control) | 0.51 | 0.51 | 0.51 | 0.51 |

At 1% false-positive rate the best signal reaches only 4.7% true positives. In 200-step
windows the separation is clean (clean 0.41-0.43 vs corrupt 0.61), so the region detects a
*rate* change but not a single event.

Root cause is capacity, not scoring: 28,321 of 32,768 possible segments are in use, so the
region is saturated and the clean burst floor stays at ~0.41.

Note: burst fraction, prediction error and unexpectedness are **provably identical** here -
`step()` only scores the input columns, so all three are monotone functions of the same
count. Reporting three "different" signals was a measurement illusion.

## D33 · Belief revision works by out-voting; deep reasoning needs isolated matrices

`tools/curriculum.cpp` extended with revision and hop-depth experiments (100 facts, k=24).

Revision of "gulab isa ped" (5 reps) by "gulab isa patthar" (m reps):

| m | recall old | recall new | recoverable | control entity |
|---|---|---|---|---|
| 0 | 1.000 | 0.000 | yes | 1.000 |
| 2 | 1.000 | 0.000 | yes | 1.000 |
| 5 | 0.125 | 0.875 | yes | 1.000 |
| 10 | 0.000 | 1.000 | no | 1.000 |
| 20 | 0.000 | 1.000 | no | 1.000 |

Hop depth on a 5-link chain (A->B->C->D->E), shuffled control in parentheses:

| depth | isolated matrix | shared matrix | shuffled (chance 0.047) |
|---|---|---|---|
| 1 | 1.000 | 0.667 | 0.031 |
| 2 | 1.000 | 0.458 | 0.049 |
| 3 | 1.000 | 0.625 | 0.038 |
| 4 | 1.000 | 0.500 | 0.045 |

Findings:
1. Revision is **out-voting**, not subtraction: break-even at m=5, takeover at m=10, and the
   old fact is recoverable with 15 reps after m=20.
2. The matrix can hold both answers; the fixed-width read-out picks one (`k` decides). A 48-bit
   read-out shows old and new both at 1.000 simultaneously.
3. Four-hop recall is exact (1.000) when each link has its own matrix, but degrades to 0.46-0.67
   with a shared matrix - the read-out is the bottleneck, not the chaining.
4. Entity codes that share bits leak: with feature bundles the control entity drops to 0.042 at
   m=20, because revision is local in weights but not in the read-out.

## D34 · Retrieval without a context window: age-invariant, load-bound

`tools/contextfree.cpp` asks whether a fixed-size associative memory can serve information
learned arbitrarily far back, with no window.

| test | result |
|---|---|
| capacity (1024x1024, k=24) | 1.000 @100, 0.970 @1k, 0.515 @5k, 0.183 @20k |
| age curve (1500 items, one learn each) | 0.89-0.91 from age 0 to 1024 |
| window 64 / 256 at the same ages | 0.000 beyond the window |
| 5-link chain, 100 chains + 500 distractors | 1.00 / 0.75 / 0.09 / 0.02 / 0.00 |
| old item used in a new role context | 0.692 (wrong key 0.054, b retrieval 1.000) |
| shared role code in one matrix | 0.26 (own background swamps the signal) |

Findings:

1. **No window is needed for single items.** Recall is flat across 1024 steps of age at fixed
   load; the cost of distance is paid in load, not in time. State size is constant (1 MB).
2. **Iterated retrieval is the real limit.** Per-hop recall 0.90, but 5 hops give 0.00 -
   errors compound super-linearly (independent errors would give 0.59). The read-out, not the
   chaining, is the bottleneck.
3. **A new context needs its own read-out.** Reusing a shared role code inside one matrix
   adds a background larger than the signal (0.26 end-to-end); one read-out per context gives
   0.69 with the same retrieval accuracy.
4. Control: shuffled values 0.032, untrained matrix 0.032 - the effect is not an artifact.

## D35 · Cleanup memory fixes the compounding error of iterated retrieval

D34 left one failure: 5-hop chains end at 0.00 because each retrieved code is approximate.
`include/bm/cleanup.hpp` adds a Kanerva cleanup memory: store canonical codes as bitsets, and
snap any approximate code to the closest one before re-using it.

| depth | raw | + cleanup |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 0.750 | 0.990 |
| 3 | 0.090 | 0.990 |
| 4 | 0.020 | 0.990 |
| 5 | 0.000 | 0.990 |

| property | value |
|---|---|
| denoise exact, 0-12 flips of 16 | 1.000 |
| denoise exact, 14 flips | 0.000 (closer to another item) |
| foreign code rejected | yes |
| codes stored / RAM | 2600 / 512 KB |
| latency per clean (4096 items) | 29.7 us |

Findings:

1. The bottleneck was never the chaining - it was the noisy hand-off between hops. Cleanup
   removes it: 495 of 500 retrieved codes were corrected, and every correction was right.
2. The mechanism is pure nearest-neighbour over bitsets: no training, no decay, fixed size
   (`items * n / 8` bytes), and it composes with any read-out.
3. Cleanup is a *closed-world* operation: it can only snap to codes it has seen. At 14 flips
   of 16 the nearest neighbour is a different item and the answer is wrong, so the threshold
   (0.6 overlap) is what makes it safe.
4. `-mpopcnt` matters: `__builtin_popcountll` without it calls libgcc (537 us per clean);
   with it the same loop is 29.7 us (18x).

### D36 · A super-linear read-out does not buy capacity

Krotov & Hopfield (2016) show the dense associative memory family gains capacity as the
read-out non-linearity grows beyond linear. The naive transplant was tested here: keep the
same local Hebbian training and replace `s[j] = sum_i W[i][j]` with `s[j] = sum_i W[i][j]^p`
for p = 1, 2, 3 (`tools/powermem.cpp`, 1024-dim codes, 24 active bits).

| load (pairs) | p=1 | p=2 | p=3 |
|---|---|---|---|
| 500 | 0.999 | 0.922 | 0.685 |
| 1000 | 0.969 | 0.786 | 0.570 |
| 2000 | 0.830 | 0.626 | 0.443 |
| 5000 | 0.512 | 0.407 | 0.308 |
| 10000 | 0.298 | 0.258 | 0.215 |
| 20000 | 0.188 | 0.175 | 0.159 |

| case | p=1 | p=2 | p=3 |
|---|---|---|---|
| shared 8-bit role tag (200 pairs) | 0.213 | 0.053 | 0.053 |
| A then B, recall A | 1.000 -> 1.000 | 1.000 -> 0.998 | 0.981 -> 0.865 |
| shuffled-value control | 0.025 | 0.025 | 0.028 |

Findings:

1. The linear sum is already the optimal matched filter for this data: squaring or cubing
   amplifies the few strongest (noisier) synapses and discards the mass of weak-but-
   consistent evidence. Every load and every control is worse.
2. The shared-context case is the worst hit (0.213 -> 0.053): the role tag contributes a
   constant background to every pattern, and the linear sum tolerates it, while a
   super-linear read-out lets that constant dominate.
3. The capacity gain in the dense-associative-memory literature is not obtained by powering
   the weight matrix. It needs the energy applied to the *overlap with each stored pattern*
   plus iterative retrieval dynamics, i.e. a different read-out, not a different exponent.
4. Consequence for CortexMesh: keep the linear read-out, and get capacity from the
   structure above it (cleanup memory, hierarchies of regions), not from a read-out trick.

### D37 · Cleanup re-ranking does not help word prediction

The cleanup memory fixes noisy codes (D35), so it was wired into the language model as a second
decoder: take the top-K predicted bits as a code, snap it to the nearest vocabulary code, and
re-rank candidates by `(1 - lambda) * linear_score + lambda * code_overlap`
(`tools/streamlm.cpp`, decode = 1, wikitext2, V = 3000, 50k tokens, identical weights).

| decoder | top-1 | top-3 | top-10 | MRR |
|---|---|---|---|---|
| linear bit-score | 0.0315 | 0.0481 | 0.0666 | 0.0440 |
| + cleanup, lambda = 0.5 | 0.0272 | 0.0412 | 0.0547 | 0.0374 |
| + cleanup, lambda = 1.0 | 0.0126 | 0.0220 | 0.0425 | 0.0383 |

Findings:

1. Snapping the predicted code to a stored code destroys information. The linear read-out is a
   *weighted* vote over bits; the snapped code keeps only which bits survived, so a word with
   three strong bits and twenty weak ones loses to a word that merely shares more bits with the
   snapped pattern.
2. The negative result is informative about where cleanup belongs: it repairs a *discrete* code
   that is already close to a stored one (hops, pose codes, hippocampal re-cue), not a *graded*
   prediction that is still being formed.
3. The earlier hop result (0.00 raw -> 0.99 cleaned) and this result are consistent: cleanup
   helps when the input is one bit-flip away from a stored code, and hurts when the input is a
   sum of evidence.

### D38 · Local prediction is easy, sustained generation is not

The character model on real Python (573 stdlib files, 585k held-out characters) was scored two
ways: next-character accuracy, and greedy continuation of a 16-character window from 6,036
sampled positions.

| model | next-char top-1 | exact 16-char continuation | mean correct prefix |
|---|---|---|---|
| N=256, ctx=8 | 0.3758 | — | — |
| N=512, ctx=8 | 0.3919 | 0.0010 | 0.725 chars |
| N=512, ctx=12 | 0.3846 | 0.0008 | 0.881 chars |

Findings:

1. Per-character accuracy near 0.39 coexists with a 16-character exact-match rate near 0.001.
   The gap is error compounding: one wrong character moves the context off the true path, and
   the model does not recover.
2. The mean correct prefix is below one character even though the first character is right 39 %
   of the time. Most continuations die on the second character, which is the signature of a
   model that has learned the *marginal* statistics of the next character but not the joint
   structure of a line.
3. Widening the code (N = 256 -> 512) buys +0.016 top-1 and nothing else; lengthening context
   (8 -> 12) buys nothing at N = 512. Both knobs are exhausted.
4. What would actually move the continuation number is search over the sequence (beam or
   neighbourhood re-scoring), not a better next-character distribution. That is the next
   experiment, and it is exactly what the cleanup memory does for discrete codes (D35).

### D39 · Beam search does not repair the continuation

Beam search over the same character model (width 4 and 8, cumulative linear score, no length
penalty) was compared with greedy decoding on the 16-character continuation task.

| decoder | exact 16-char | mean correct prefix |
|---|---|---|
| greedy | 0.0010 | 0.725 |
| beam width 4 | 0.0010 | 0.647 |
| beam width 8 | 0.0010 | 0.620 |

Findings:

1. Search makes it slightly worse, not better. The read-out score is a sum of uncalibrated
   linear activations, so the beam prefers paths through high-frequency characters (spaces,
   newlines) that accumulate score without being likely.
2. The greedy path is already close to the score-optimal path; the problem is not search, it
   is that the model's *next-character distribution is only weakly conditioned* on the line so
   far. Widening the beam cannot add information the model does not have.
3. Consequence: the fix has to be representational (a context that carries more of the line:
   syntax, indentation, the enclosing block), not procedural (better search).

### D40 · Word-level is the wrong granularity for code

The same streaming model was run on the Python corpus in word mode (`tools/streamlm.cpp`,
V = 5000, 1-word context) instead of character mode.

| granularity | train inv_top1 | held-out transfer inv_top1 |
|---|---|---|
| word (V = 5000) | 0.0499 | 0.0521 |
| character (N = 512, ctx = 8) | 0.3919 | — |

Findings:

1. Code has 24,354 distinct tokens in 795,062 tokens (3.1 %), versus about 1.5 % for prose. The
   vocabulary is both larger and flatter, so a word-level model spends its capacity on
   identifiers it sees once.
2. Word-level prediction on code (0.050) is *worse* than word-level prediction on prose (0.10),
   even though code is far more predictable to a human. The structure that makes code predictable
   (indentation, matching brackets, short identifiers) is sub-word.
3. The character model wins by 8x on the same corpus. For code, the character is the right atom.

### D41 · The Hebbian matrix saturates, and more data makes it worse

The full-corpus run (86.5M tokens, V = 5000, 3-word context, xi = 0.005) was supposed to be the
headline result. It collapsed:

| tokens seen | top-1 | inv_top1 |
|---|---|---|
| first half (43.2M) | 0.0065 | 0.0065 |
| second half (43.2M) | 0.0000 | 0.0000 |
| whole corpus (86.5M) | 0.0033 | 0.0039 |
| held-out valid | 0.0000 | 0.0000 |

The same configuration scored 0.0760 in-vocabulary top-1 at 2M tokens. A direct measurement of
the weight matrix after 8M tokens shows why:

| xi | mean weight | frac > 0.9 | frac > 0.5 | inv_top1 |
|---|---|---|---|---|
| 0.005 | 0.9902 | 0.9694 | 0.9996 | 0.0368 |
| 0.0005 | 0.7797 | 0.5228 | 0.8007 | 0.0808 |

Findings:

1. The update `w += xi * g * (1 - w)` drives every synapse that is ever co-active toward 1. After
   8M tokens, 97 % of the matrix is above 0.9; the read-out can no longer separate one context
   from another, so prediction falls to chance.
2. More data is actively harmful in this regime: the model is best at an intermediate amount of
   experience, which is the signature of a system with no forgetting.
3. The learning rate is a *capacity* knob, not just a speed knob: lowering it from 0.005 to 0.0005
   more than doubles in-vocabulary top-1 at the same token count (0.0368 -> 0.0808) purely by
   delaying saturation.
4. The fix is structural, not parametric: the matrix needs forgetting proportional to activity
   (decay on every touched row), or a cap on how much any synapse can absorb, or sparse
   connectivity with a fixed budget per column. A 0.99999 per-batch decay is four orders of
   magnitude too weak to matter at 86M tokens.
5. This is the same failure mode that made the second epoch worse than the first (top-1 0.0637 ->
   0.0558): repetition is just faster saturation.

### D42 · Importance-scaled learning fixes the saturation

The research review (docs/34) said the positive-only accumulator has no fixed point below 1. Two
replacement rules were implemented in `tools/streamlm.cpp` (rule = 1, 2):

| rule | update | fixed point |
|---|---|---|
| 0 plain | `w += xi*g*(1-w)` | 1 (saturates) |
| 1 importance | `dw = xi*g/(1+lam*omega)`, `omega += |dw|` | below 1, per synapse |
| 2 covariance | `w += xi*g*((1-mx)(1-my) - w)` | `(1-mx)(1-my)` |

Measured at 3M tokens, wikitext-103, V=5000, 3-word context, xi=0.005:

| rule | inv_top1 | top3 | mean w | frac > 0.9 |
|---|---|---|---|---|
| 0 plain | 0.0714 | 0.0702 | 0.9433 | 0.8237 |
| 1 importance (lam=10) | **0.0849** | 0.0772 | 0.7401 | 0.3609 |
| 2 covariance | 0.0718 | 0.0705 | 0.8914 | 0.7654 |

Findings:

1. The importance-scaled rule wins by +19 % in-vocabulary top-1 at the same token count, and cuts
   the saturated fraction of the matrix from 82 % to 36 %. This is the SI/MAS analogue from the
   continual-learning literature, and it needs one extra float per synapse (12.6 MB at ctx=3).
2. Mean subtraction (rule 2) does *not* help on the real corpus, because the DC mode is not the
   binding constraint here: the binding constraint is that the same pair is reinforced tens of
   thousands of times. The synthetic Zipf test in `tools/learning_rule.cpp` confirmed that no rule
   collapses when each pair is trained once.
3. The remaining saturated 36 % is the next target: a per-synapse cap, or column renormalisation,
   should push it lower without losing the frequent-word structure.

## D43 · A hard cap on Hebbian accumulation destroys the ranking

Measured on the same 3M-token run, `rule=3` = importance-scaled Hebb with a hard ceiling
`w = min(w + dw, wcap)`, `wcap = 0.6`, `lam = 10`.

| rule | inv_top1 | top3 | mean w | frac > 0.9 | frac at cap |
|---|---|---|---|---|---|
| 1 importance | 0.0849 | 0.0772 | 0.7401 | 0.3609 | - |
| 3 importance + hard cap 0.6 | **0.0272** | 0.0272 | 0.5643 | 0.0000 | **0.7526** |

Findings:

1. Capping removes saturation but creates a worse failure: 75 % of synapses are pinned to the same
   value, so they tie, and the tie is broken by the frequency normaliser rather than by content.
   Top-1 fell from 0.0849 to 0.0272, a 3.1x regression.
2. The lesson is that the *shape* of the ceiling matters more than its height: the update must slow
   down as it approaches the ceiling (`dw ~ 1 - w/wcap`) instead of being clipped. Rule 4 in
   `tools/streamlm.cpp` implements that and is measured in the next entry.
3. This also rules out column renormalisation as a "free" fix: any hard projection that makes many
   entries equal is harmful. Renormalisation must preserve the ratios.

## D44 · Every ceiling we tried loses to no ceiling (at 3M tokens)

| rule | inv_top1 | top3 | mean w | frac > 0.9 | frac at cap |
|---|---|---|---|---|---|
| 1 importance (no ceiling) | **0.0849** | 0.0772 | 0.7401 | 0.3609 | - |
| 3 importance + hard cap 0.6 | 0.0272 | 0.0272 | 0.5643 | 0.0000 | 0.7526 |
| 4 importance + soft target 0.7 | 0.0732 | 0.0701 | 0.6002 | 0.0000 | 0.0000 |
| 5 importance + ratio-preserving column renorm (0.7) | 0.0765 | 0.0726 | 0.7167 | 0.3527 | 0.5894 |

Findings:

1. Removing saturation and improving accuracy are not the same objective. At 3M tokens the plain
   importance rule is still the best of the four, even though 36 % of it is above 0.9.
2. The reason every ceiling loses is the same: once many synapses share a value, the frequency
   normaliser (`1/(freq_bit+0.05)`) decides the ranking instead of the content. Rule 5 was designed
   to avoid this, and it does preserve within-column ratios, but a column whose entries are all
   near-equal still collapses to all-at-cap (59 % at cap here).
3. The correct next move is therefore not a better ceiling but a non-saturating memory: a fixed-size
   integer context cache with LRU, which does not accumulate at all. Tunable ceilings are a
   dead end; the matrix is the wrong substrate for 86M tokens.
4. The one solid win stands: importance scaling alone, 8M tokens, 0.0368 -> 0.0806 in-vocabulary
   top-1 (+119 %), saturation 97 % -> 61 %.

## D45 · Synaptic homeostasis (SHY) removes the saturation but does not buy accuracy at 8M

`rule=6` = importance-scaled Hebb plus the measured biological downscaling: every 400k tokens the
weakest 80% of synapses are scaled by 0.82, the top 20% are left alone (de Vivo et al. 2017, Science,
doi:10.1126/science.aah5982: axon-spine interface -18% after sleep vs wake, proportional to size).

| corpus | rule | inv_top1 | mean w | frac > 0.9 |
|---|---|---|---|---|
| 3M | 1 importance only | **0.0849** | 0.7401 | 0.3609 |
| 3M | 6 importance + SHY | 0.0765 | 0.2363 | 0.2000 |
| 8M | 1 importance only | **0.0806** | 0.8713 | 0.6084 |
| 8M | 6 importance + SHY | 0.0744 | 0.2442 | 0.2000 |
| 3M | 5 column renorm | 0.0765 | 0.7167 | 0.3527 |
| 3M | 4 soft target | 0.0732 | 0.6002 | 0.0000 |

Findings:

1. SHY does what it says on the tin: it cuts the saturated fraction from 36% to 20% and drops the mean
   weight from 0.74 to 0.24, i.e. it really does remove mass from the tail. It does so at a small
   accuracy cost (0.0849 -> 0.0765, -10%).
2. The 8M row confirms it at the scale where the plain rule starts to decline: SHY cuts the saturated
   fraction from 61% to 20% and still costs 8% accuracy. Combined with D43/D44, the conclusion is
   solid: at 3M-8M tokens **saturation is not the binding constraint**. Every mechanism that reduces it also reduces accuracy, because the pairs that saturate
   are the frequent ones that carry the actual predictive signal.
3. The matrix is therefore not the bottleneck at this scale. The bottleneck is the representation
   (D36, D39, D40): a 512-bit code per word with a linear sum over 3 context rows cannot express the
   conditional distribution, no matter how the weights are shaped. Fixing the update rule cannot fix
   that.
4. SHY stays in the tool as the mechanism for long horizons (86.5M) and for the sleep consolidation
   loop (docs/37), where it is the only rule with a measured biological effect size.

## D46 · Age-of-acquisition measurement is now available (`aoa`)

`streamlm ... <aoa flag>` records the token index at which each word is first predicted in the top 5,
then reports the median first-hit per frequency decile and the Pearson correlation between log corpus
frequency and log AoA.

At 300k tokens: `aoa_pearson_logfreq_logaoa = -0.4292` over the 25 words acquired so far. Frequent
words are acquired earlier, as child data predicts. This is the first measurement in the project that
can distinguish a child-like learner from an n-gram: Chang & Bergen (2022) show LMs lean on frequency
far more than children do, and Frank (2026) show children's learning accelerates with exposure.

## D47 · A hashed n-gram cache beats the Hebbian matrix on every metric at 8M tokens

`tools/cachelm.cpp` + `include/bm/ngram_cache.hpp`: open-addressed hash table, 64-bit context hash,
up to 3 continuations per context with saturating counts, coldest-slot eviction once the table is
85% full, periodic sweep. 2^21 slots x 24 B = 48 MB. Word ids are memory addresses here, never a
representation of meaning.

| system | bytes | top1 | top3 | top10 | mrr | tokens/s |
|---|---|---|---|---|---|---|
| streamlm matrix (rule 1) | 3 MB | 0.0663 | 0.0708 | 0.0830 | 0.0726 | 5,500 |
| cachelm order 5 | 48 MB | 0.0948 | 0.1369 | 0.1508 | 0.1160 | 110,000 |
| cachelm order 4 | 48 MB | 0.1001 | 0.1463 | 0.1624 | 0.1235 | 121,000 |
| cachelm order 3 | 48 MB | **0.1079** | **0.1600** | **0.1778** | **0.1344** | 135,000 |
| cachelm order 3, 1M slots | 24 MB | 0.0891 | 0.1299 | 0.1425 | 0.1095 | 129,000 |

Findings:

1. Same corpus, same vocabulary, same metric: **top-1 +63%, top-3 +126%, top-10 +114%, MRR +85%**
   over the best Hebbian rule, and 22x faster (135k vs 5.5k tokens/s).
2. Order 3 beats order 4 and order 5. Longer contexts are sparser: at 85% load most 4- and 5-grams
   are evicted before they can be reused, so the backoff weight (lambda=0.4) is applied to noise.
   The optimum is set by the table size, not by the language.
3. Memory is the real cost: 48 MB for 48 MB of value against 3 MB of weights. That is the trade the
   D45 conclusion predicted - you buy accuracy by not accumulating, and you pay in slots.
4. The cache does not saturate by construction: an entry is either present or gone, so there is no
   weight to run away. This is the "non-saturating memory" that D43-D45 said was missing.
5. Next: interpolation (matrix + cache), and a sweep at 86.5M tokens to see how much of the gain
   survives when the table thrashes.

## D48 · The cache trades memory for accuracy along a log-linear curve

Same corpus (86.5M tokens), same order (3), same 3 continuations per context; only the table size
changes. 24 bytes per slot, so the slot count is bytes/24.

| bytes | slots | top1 | top3 | top10 | mrr |
|---|---|---|---|---|---|
| 6 MB | 262,144 | 0.0761 | 0.1108 | 0.1190 | 0.0930 |
| 12 MB | 524,288 | 0.0865 | 0.1272 | 0.1379 | 0.1066 |
| 24 MB | 1,048,576 | 0.0978 | 0.1449 | 0.1589 | 0.1214 |
| 48 MB | 2,097,152 | **0.1100** | **0.1636** | **0.1816** | **0.1372** |

Findings:

1. Roughly +0.011 top-1 per doubling of the table, i.e. log-linear in memory. That is the signature
   of a capacity-limited exact memory, not a broken one: more slots means fewer evictions of useful
   contexts, and the curve flattens only when the corpus stops producing new contexts.
2. Even the 6 MB point is 19x better than the 3 MB Hebbian matrix on the same corpus (0.0761 vs
   0.0039). The matrix's problem was never memory, it was accumulation.
3. The interesting comparison is not cache-vs-matrix but cache-vs-nothing: at 86.5M tokens a 6 MB
   exact memory beats a 3 MB learning matrix by 19x. That sets the engineering direction: keep the
   matrix as the generaliser (small), spend the budget on the exact memory.

## D49 · Interpolating the matrix with the cache beats either alone (500k tokens)

`streamlm ... <cache_bits=21> <cache_order=3> <cache_beta=0.5> <cache_lambda=0.4>`. The cache scores
are normalised by their own max, then by the matrix's own top score, then blended with `cache_beta`.

| system | top1 | top3 | top10 | mrr | inv_top1 | inv_top10 |
|---|---|---|---|---|---|---|
| matrix only (rule 1) | 0.0718 | 0.0733 | 0.0854 | 0.0769 | 0.0804 | 0.0956 |
| matrix + cache | **0.0897** | **0.1127** | **0.1280** | **0.1054** | **0.1005** | **0.1433** |

Findings:

1. +25% top-1, +54% top-3, +50% top-10, +37% MRR. The two memories fail differently: the matrix
   generalises to contexts it has never seen, the cache is exact where it has seen them. Blending
   takes the union of the two strengths.
2. Cost: 2.5k-3.3k tokens/s against 5.5k for the matrix alone at this scale, because the blend scans
   the candidate list per cache candidate. That is acceptable at 500k and would be the thing to fix
   (a hash map from word to score, or merging the cache into `wscore` before ranking) before running
   8M+.
3. The 8M interpolation run was attempted and abandoned at 39 min without finishing; the scan cost
   dominates. The 500k point is the honest measurement.

## D50 · The grammar benchmark exposes a collapse that top-1 accuracy hid

`tools/grammar.cpp` scores 28 hand-written minimal pairs by forced choice: does the model score the
grammatical continuation above the ungrammatical one? Chance is 50%. Context is a set of word
positions (`offsets`), so consecutive (`1,2,3`) and skip (`1,2,3,4,6,8`) contexts can be compared.

Plain Hebb, no cache contribution (matrix only, cache on):

| tokens | rule | agreement_adj | agreement_long | determiner | aux | one_of | polarity | OVERALL |
|---|---|---|---|---|---|---|---|---|
| 2M | plain Hebb | 50.0% | 66.7% | 83.3% | 25.0% | 100% | 50.0% | 60.7% |
| 20M | plain Hebb | 12.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **3.6%** |
| 20M | importance + SHY | 50.0% | 66.7% | 66.7% | 25.0% | 100% | 50.0% | **57.1%** |
| 86.5M | importance + SHY | 50.0% | 66.7% | 66.7% | 25.0% | 100% | 50.0% | **57.1%** |

Findings:

1. **More data makes grammar worse.** 2M tokens -> 60.7%, 20M tokens -> 3.6%, i.e. far below chance.
   This is the saturation collapse of D41-D44 showing up in a place where top-1 never showed it: at
   20M the matrix is so uniformly high that the frequency normaliser, not the context, decides the
   score.
2. **Importance scaling plus SHY recovers it**: 3.6% -> 57.1%. So the update rule was not cosmetic.
   The D45 verdict ("saturation is not the binding constraint") is wrong for grammar: it is not the
   binding constraint for top-1, but it is the binding constraint for structure.
3. Still only 57.1% against 50% chance. The model has a weak preference for the right agreement, not
   a representation of agreement. Skip contexts (6 slots instead of 3) were measured at 20M and are
   in `logs/grammar_skip_20m.log`.
4. 86.5M tokens gives the same 57.1%. The curve is flat from 20M to 86.5M: the fix stops the
   collapse but nothing turns the weak preference into structure. Adding more data does not add
   grammar, and adding more context slots does not either (skip context `1,2,3,4,6,8` at 20M is also
   57.1%, see `logs/grammar_skip_20m.log`).
5. This is the first benchmark in the project that measures something the model does not do. Every
   previous benchmark measured something it does do, slightly better than a baseline.
6. Why 57% and not 50%: with a 3-word window the model can exploit shallow cues (the noun's own
   number often correlates with the verb's in the training text, e.g. "keys ... turn" as a collocation)
   without representing agreement. That is the same failure Chang & Bergen (2022) measured in LMs:
   frequency beats structure.

## D51 · Stateful (trace) context beats a fixed window, but neither reaches grammar

`tools/grammar.cpp` gained a second context mode: instead of the last 3 words at fixed offsets, each
bit keeps a decaying trace (`trace[j] = gamma*trace[j] + 1` when bit j fires), and the matrix reads
the top-k (48) trace bits. The context is a state with unbounded time depth and bounded memory, the
cheap version of an HTM-style sequence memory.

Matrix only, no cache, 20M tokens, importance + SHY:

| context | agreement_adj | agreement_long | determiner | aux | one_of | polarity | OVERALL |
|---|---|---|---|---|---|---|---|
| fixed slots 1,2,3 | 50.0% | 66.7% | 66.7% | 25.0% | 100% | 50.0% | 57.1% |
| trace k=48 gamma=0.9 | 50.0% | 66.7% | 66.7% | 50.0% | 100% | 50.0% | 60.7% |
| trace k=48 gamma=0.95 | 50.0% | 66.7% | 66.7% | 50.0% | 100% | 50.0% | 60.7% |

Findings:

1. The stateful trace is worth +3.6 points (57.1% -> 60.7%) and it is the only change so far that
   improves grammar at all. It also removes the collapse: at 20M the plain rule collapsed to 3.6%,
   the bounded rule with slots gives 57.1%, and with a trace 60.7%.
2. It is still nowhere near grammar. 60.7% against 50% chance on pairs a human gets 100% of the time
   means the model has a preference, not a rule.
3. Why it plateaus: the code is a random superposition, so "plural" is spread across all 24 bits of a
   word and shares those bits with meaning, tense and animacy. The matrix has to learn 24x24
   second-order terms per word pair through that interference. The window was never the only problem;
   the *representation* has no factor for number to attach to.
4. Next experiment, already designed: a hybrid code with explicit feature bit-ranges (number, person,
   tense) alongside the random identity bits, labelled by hand for the closed-class words in the
   minimal pairs. If agreement jumps, the fix is feature discovery; if it does not, the readout is
   the limit.

## D52 · The matrix learns the rule, the linear readout cannot use it

D51 left one open question: does a hybrid code (random identity bits + explicit feature bit-ranges)
make agreement learnable? It does, and the failure moves somewhere else entirely.

`tools/grammar.cpp` gained `code_mode=2` (features from a hand table) and `code_mode=3` (features
from morphology, so a novel noun gets a number feature without being listed), with one bit per
feature repeated `feat_repeat` times, and the readout averaging over a word's active bits rather than
summing.

### The rule is in the weights

Direct measurement of the 16x16 feature blocks after training (env `GRAMMAR_BLOCKS`), 1.5M tokens of
a deterministic-agreement corpus, xi=5e-4:

| block | value |
|---|---|
| PLURAL x VERBBASE | 0.880 |
| PLURAL x VERB3SG | 0.519 |
| SINGULAR x VERB3SG | 0.662 |
| SINGULAR x VERBBASE | 0.566 |

Both directions are correct and the margin is 0.36, i.e. the matrix has learned a genuine
number-agreement rule, not a preference. That is the first time this architecture has been shown to
contain a grammatical rule in a readable form.

### The task still fails

Forced-choice accuracy on the same corpus, 22 scorable pairs:

| configuration | OVERALL |
|---|---|
| xi=1e-3 (saturated) | 40.9% |
| xi=5e-4, boost=1 | 45.5% |
| xi=1e-4, boost=1 | 59.1% |
| xi=1e-4, boost=4 | 63.6% |
| xi=1e-4, boost=16 | 63.6% |
| xi=1e-4, feat-only scoring | 54.5% |
| xi=1e-4, no mean subtraction | 36.4% |

Findings:

1. Saturation destroys the rule. At xi=1e-3 every block sits at 0.999 and agreement accuracy is
   *below* chance (40.9%). The rule only exists in the linear regime, which bounds xi from above by
   the number of times a feature pair can co-occur.
2. The readout is the binding constraint, not the learning. The blocks are right by 0.36, yet the
   per-pair margins are ~2-8 points against a ~7-point global bias toward the more frequent form.
   The feature differential and the identity bias are the same order of magnitude, so they cancel.
3. Scoring feature bits alone does not rescue it (54.5%): the residual is then a bias created by the
   per-bit mean subtraction, which is not constant across feature groups. Removing the mean
   subtraction is worse still (36.4%), because then the identity bits dominate outright.
4. Boosting the feature learning rate is self-defeating: at boost>=4 the feature blocks saturate and
   the ratio that made them informative disappears. Features must stay *slower* than identity bits,
   not faster.
5. Widening the window is necessary but not sufficient. Offsets 1,2 (36.4%) < 4,5,6 (45.5%) <
   1..6 (59.1%): the subject must be inside the window, or PLURAL never enters the context at all.
6. The residual errors are agreement attraction. "the box that the children opened" is answered
   "were" because "children" is nearer than "box". A readout that sums over context bits with equal
   weight cannot prefer a clause head over a nearer noun; that needs structure (a head-dependent
   term, a stack, or a multiplicative binding), not more capacity.

### What this means for the design

The linear second-order readout has a hard ceiling near chance plus a few points on forced choice,
regardless of whether the representation carries the factor. Two consequences:

- Keep the hybrid feature code. It is the only representation change that produced a correct rule
  (0.36 margin) rather than a preference, and it is the natural target for feature discovery.
- The next real change must be in the readout, not the code: a competitive or normalising term
  (softmax over candidates, column normalisation applied *per readout*, or a multiplicative
  context-word binding) is required before more capacity or more data can help.

Reproduction: `python3 tools/gen_agree.py 1 40000 data/agree_train.txt` then
`GRAMMAR_BLOCKS=1 ./build/bin/grammar data/agree_train.txt 10000 1500000 1,2,3,4,5,6 0.0001 1 0.5 10 0 48 0.9 3 16 0.05 1`.

## D53 · The matrix does exhibit agreement; it applies it to the nearest noun

D52 ended with "the readout is the binding constraint". The fix is not a fancier readout. It is to
stop letting word identity compete with the feature under test, and to stop scoring pairs whose
context the model cannot even represent.

Two changes to `tools/grammar.cpp`:

1. `code_mode=4`: a **feature-only code**. Identity bits are dropped for any word that has a
   feature, so "turn" and "turns" differ only in VERBBASE vs VERB3SG and nothing else. Words with no
   feature keep a 4-bit code so they still occupy a position.
2. A pair is **not scorable** if any word in its context is unknown to the model. Such a word
   becomes UNK, loses its feature, and the item degenerates into a coin flip decided by the global
   bias — counting it inflates the error rate for a reason unrelated to grammar.

`tools/gen_agree.py` was rewritten to generate the training corpus from the *same lexicon* as the
pairs (keys/boxes/children/men/teams/..., that/who, one of the, in/on/with/near). The pairs
themselves are never emitted, so the test remains generalisation to novel combinations of known
words. Corpus: 60k sentences, every one grammatical, with relative-clause distractors and `one of
the X` cases.

### Result

Forced choice, 28 pairs, xi=1e-4, hybrid feature code, 3M tokens:

| split | n | ok | acc |
|---|---|---|---|
| unambiguous (one number-bearing word in context) | 8 | 8 | **100.0%** |
| ambiguous (two number-bearing words that disagree) | 6 | 1 | 16.7% |

Every pair whose context carries a single number is answered correctly. Of the six pairs whose
context carries two conflicting numbers, five are answered wrongly and one correctly. That is not
noise: it is a single mechanism.

### The mechanism

The model has learned number agreement and applies it to the **nearest** number-bearing word:

- "the keys that the man found" -> answered *turns*, following "man" (singular, nearer) instead of
  "keys" (plural, subject).
- "the box that the children opened" -> answered *were*, following "children".

Both are the classic agreement-attraction error, and they are the mirror image of the correct cases.
The readout sums over context bits with equal weight, so the closest feature-bearing word dominates
the sum. This is a property of a bag-of-features readout, not a failure of learning.

### Why the competitive readout did not help

`readout_mode=1` centres the two candidates' bit activations on the mean over the *union* of their
bit sets. Algebraically that is `sum_Sg(b) - sum_Sb(b) - c(|Sg| - |Sb|)`: when the two codes have the
same size the centre cancels exactly, so it is equivalent to no centring at all (36.4% either way).
`readout_mode=2` (softmax) inherits the same scores. A global offset shared by two equal-sized
codes is invisible to any comparison of their sums; the bias that hurts is *per-word*, and the only
way to remove it is to remove word identity from the code, which is what `code_mode=4` does.

### Why feature boosting is self-defeating

Boosting the feature-bit learning rate (`feat_boost>=4`) looks like the obvious fix for "features
must outweigh identity". It inverts: the feature blocks saturate at 0.999 and the 0.36 margin that
made them informative disappears, dropping accuracy from 63.6% to 45.5%. Features must learn
*slower* than identity bits so they stay in the linear regime where co-occurrence counts stay
proportional to weight.

### What this changes

D52's conclusion was "the readout is the limit, the code is fine". That is half right. The readout is
the limit *for ambiguous contexts*, and the code is fine for unambiguous ones (100%). The remaining
gap on real text is therefore not capacity or data: it is that the model has no notion of *which*
noun is the subject. Any further progress needs a structural cue — clause-head detection, a stack, or
a multiplicative binding between a head marker and its dependents — not a better linear readout.

Wikitext at 20M tokens is ~40 minutes per configuration on this machine (1M tokens = 122 s), so the
wikitext comparison of `readout_mode` 0 vs 1 is running separately rather than being reported here.

Reproduction:
`python3 tools/gen_agree.py 1 60000 data/agree_train.txt` then
`./build/bin/grammar data/agree_train.txt 10000 3000000 1,2,3,4,5,6 0.0001 1 0.5 10 0 48 0.9 4 8 0.05 1 0 1.0`.

## D54 · The slot matrices were vestigial: all slots received the same context

D53 closed with "the model has no notion of which noun is the subject". The first half of that was
wrong, and the reason is a bug rather than a limitation of the architecture.

`tools/grammar.cpp` keeps one N x N matrix per context slot, and `score_bits()` reads slot `s` using
only the bits of the word at `offsets[s]` — correct. Training did not match:

```cpp
ctx_idx.clear();
context_indices(ctx_idx);          // union of the bits of EVERY context word
for (size_t s = 0; s < n_slot; ++s) {
  float* row = w.data() + s * N * N;
  for (uint32_t i : ctx_idx) { ... }   // same union fed to every slot
```

So every slot received identical updates and the per-slot matrices were identical copies. Position
carried no information at all: the model was a bag of words over a window, with six redundant copies
of one matrix. This was invisible until the blocks were measured per slot:

| slot | PL x BASE | PL x 3SG | SG x 3SG | SG x BASE |
|---|---|---|---|---|
| before fix (all slots) | 0.6112 | 0.4750 | 0.5048 | 0.3074 |
| after fix, slot0 (distance 1) | 0.4123 | 0.2382 | 0.3309 | 0.0948 |
| after fix, slot2 (distance 3) | 0.0884 | 0.0910 | 0.0000 | 0.0000 |
| after fix, slot3 (distance 4) | 0.3141 | 0.2276 | 0.2442 | 0.1651 |
| after fix, slot4 (distance 5) | 0.2839 | 0.1619 | 0.2325 | 0.1180 |
| after fix, slot5 (distance 6) | 0.1036 | 0.1477 | 0.0438 | 0.0000 |

### Effect

Forced choice, 3M tokens of the lexicon-matched corpus, 6 slots:

| | unambiguous | ambiguous | OVERALL |
|---|---|---|---|
| before fix | 8/8 = 100.0% | 1/6 = 16.7% | 64.3% |
| after fix | 7/8 = 87.5% | 3/6 = 50.0% | **71.4%** |

The ambiguous pairs — agreement at a distance — improved from 16.7% to 50.0%. That is the first
change that moves the hard half of the task.

### Why it works

After the fix each slot has its own association. In "the keys that the man found turn" the verb sits
at distance 4, so it reads slot3, where the model learned PLURAL -> base form (0.314) but never
learned a verb association for slot2 ("man" is only ever followed by "found" in training, so
SG x * at slot2 is 0.0000). The distractor contributes nothing because it has no learned
verb association in its own slot, and the subject wins. The model is now sensitive to *where* a word
sits, which is enough to prefer the subject over a nearer noun in half the attraction cases.

### What is still broken

Truncating the relative clause at evaluation time (`ctx_mode=2`) changes nothing (71.4% either way).
The reason is now clear and is a genuine architectural limit, not a bug: the association
"keys -> turn" exists only in the slot that matches the distance at which it was trained. Moving the
subject to distance 1 puts it in slot0, where the model only ever saw "keys -> that". There is no
position-independent subject-verb rule anywhere in the matrix; there are six distance-specific ones.
Removing the intervening material therefore does not help, because it moves the cue into a slot with
no learned association.

Three slots is not enough either: 35.7% overall, because the subject at distance 4 falls outside the
window entirely.

### Where this leaves the design

The representation is not the bottleneck (D53: 100% on unambiguous pairs), and the window is not the
bottleneck (6 slots, subject included). The missing piece is **translation invariance**: the matrix
must be able to bind a subject to a verb regardless of how much material sits between them. Two
concrete options, both consistent with the local-learning constraint:

1. Content-addressable binding: bind the subject's code to the verb's code through a circular
   convolution / permutation, so the association is stored relative to the *item* rather than the
   *position*. Reading it back is a correlation, which is already O(N) here.
2. A head stack: on a relativiser, push the current head; the verb reads from the top of the stack
   instead of from a distance-indexed slot. This is the cheap structured-memory version and it is
   what the ctx_mode=2 experiment was approximating.

Option 2 is the smaller change and directly targets the failure mode measured here.

## D55 · Position information is worth having only when the data can pay for it

D54's fix (per-slot training) helped on the lexicon-matched synthetic corpus and hurt on wikitext.
Both directions are real and they measure the same tradeoff.

Same configuration on both corpora: 20M tokens of wikitext, 3M of the synthetic lexicon corpus,
6 slots, hybrid feature code, xi=1e-4, no cache.

| corpus | context | unambiguous | ambiguous | OVERALL |
|---|---|---|---|---|
| synthetic | slots only (D54) | 7/8 = 87.5% | 3/6 = 50.0% | **71.4%** |
| synthetic | slots only, before the slot fix | 8/8 = 100.0% | 1/6 = 16.7% | 64.3% |
| wikitext | slots only, after the slot fix | 11/20 = 55.0% | 2/8 = 25.0% | 46.4% |
| wikitext | union in every slot (before the fix) | 11/20 = 55.0% | 5/8 = 62.5% | **57.1%** |

### Reading it

1. On the synthetic corpus the slots carry a clean rule and there are only 53 word types, so each
   slot sees tens of thousands of examples. Per-slot training is affordable and position pays:
   64.3% -> 71.4%.
2. On wikitext the vocabulary is 10k and each slot sees one sixth of the data. Every slot is
   starved, and the pooled context that the old code accidentally provided (the union fed to every
   slot, i.e. six copies of one bag matrix) is a stronger predictor than six undertrained
   position-specific ones: 57.1% vs 46.4%.
3. The unambiguous half does not move on wikitext (55.0% either way). What moves is the ambiguous
   half, and it moves in opposite directions: the pooled bag is better at real-text agreement at a
   distance, the slots are better when the rule is clean and the lexicon is tiny.

### The hybrid does not rescue it

A third mode was added: `ctx_mode=2` keeps the per-slot matrices and adds one extra "bag" matrix
that sees the union of the context, with the slot contribution scaled by `pos_w`.

| ctx_mode | pos_w | unambiguous | ambiguous | OVERALL |
|---|---|---|---|---|
| 0 (slots only) | - | 7/8 = 87.5% | 3/6 = 50.0% | **71.4%** |
| 2 (bag + slots) | 0.5 / 1.0 / 2.0 | 8/8 = 100.0% | 0/6 = 0.0% | 57.1% |

On the synthetic corpus the hybrid is worse than slots alone at every weight tried: the bag has no
position, so it contributes a pooled vote that is right for the unambiguous pairs (where the subject
is also the nearest noun) and actively wrong for the ambiguous ones (where it averages the subject
and the distractor). The two signals are not complementary here; the bag dilutes the one thing the
slots were adding.

### Conclusion

Position information is not free. It costs a factor of `n_slot` in data per parameter, and on real
text that cost exceeds the benefit. Two coherent positions follow:

- If the target is real text, the pooled context should stay, and the per-slot structure should be
  introduced only where there is enough data per slot to learn it — for example by sharing the slot
  matrices across slots and learning a single position-decay function instead of six independent
  matrices. That keeps the parameter count flat while still distinguishing "one word back" from
  "four words back".
- If the target is a clean rule, per-slot is right and the synthetic result (71.4%) stands.

The cheap version of the first option — one shared matrix plus a scalar decay `gamma^s` per slot —
is the obvious next experiment: it has the same parameter count as the pooled bag, so it cannot
starve, and it still gives the model a notion of distance.

## D56 · A head stack fixes agreement at a distance; the other phenomena need other cues

D54 identified the missing piece as translation invariance and named two options. The cheap one —
a head stack — is implemented in `tools/grammar.cpp` (`head_mode`) and it works on exactly the
failure it targets.

Mechanism. On a relativiser (`that`/`who`/`which`) the most recent noun is pushed onto a bounded
stack; the verb that closes the clause reads the stack top. The association is learned into one
extra N x N matrix (`w_head`), so it costs N^2 parameters regardless of how many heads exist, and it
binds head to verb by identity rather than by distance. `head_w` scales its contribution to the
readout.

Lexicon fix that was needed first: the relative-clause verbs in the minimal pairs (`found`, `opened`,
`saw`) and the nouns `men`/`women` were absent from the generated corpus, so all four
agreement-at-a-distance pairs were unscorable and the head stack had nothing to act on. Extending
`tools/gen_agree.py` raised the scorable set from 14 to 20 pairs.

### Result

Forced choice, 3M tokens, 6 slots, hybrid feature code, xi=1e-4:

| phenomenon | n | without head stack | with head stack (head_w=4) |
|---|---|---|---|
| agreement_adjacent | 8 | 75.0% | 75.0% |
| agreement_long | 6 | 66.7% | **83.3%** |
| determiner_noun | 6 | 33.3% | 33.3% |
| aux_agreement | 4 | 50.0% | 50.0% |
| one_of | 2 | 0.0% | 0.0% |
| polarity | 2 | 0.0% | 0.0% |
| **unambiguous** | 12 | 83.3% | **91.7%** |
| **ambiguous** | 8 | 50.0% | 50.0% |
| **OVERALL** | 20 | 70.0% | **75.0%** |

Per-pair detail for the relative-clause cases, which are the ones the stack is for:

| context | correct | head | head score (good/bad) |
|---|---|---|---|
| the keys that the man found | turn | 35 | 7.996 / 5.099 |
| the key that the man found | turns | 32 | 6.984 / 3.061 |
| the boxes that the children opened | were | 42 | 7.996 / 3.272 |
| the box that the children opened | was | 25 | 5.197 / 3.061 |
| the men who the women saw | left | 26 | 7.996 / 5.099 |
| the man who the women saw | left | 29 | 3.061 / 6.984 |

Five of six. The failure is the last one, where the embedded subject `women` is itself plural and
the head is singular: the model follows the nearer noun. That is agreement attraction again, one
level down.

### What the head stack does not touch

The remaining errors are all different phenomena with different cues, and no single mechanism will
fix them together:

1. `determiner_noun` (2/6): "this car" vs "these cars". The determiner and the noun's own
   morphology disagree, and the correct answer requires the determiner to override the noun. A
   bag-of-features readout has no override; it needs a dominance relation.
2. `one_of` (0/2): "one of the keys" takes a singular verb. The plural noun is *inside* a
   quantified phrase, so the head is "one", not "keys". The stack pushes the nearest noun, which is
   the wrong head; it would need to know that `one of` forms a constituent.
3. `aux_agreement` (2/4): the two failing pairs use `he` and `they`, which are not in the generated
   corpus at all, so they score as a coin flip. The features exist (D53 added them); the corpus does
   not.
4. `polarity` (0/2): negation scope. Nothing in the current architecture represents scope.

### Where the design stands

Forced-choice agreement has gone from 57.1% (D51, trace, random codes) to 60.7% (D52, importance +
SHY) to 70.0% (D54, per-slot) to **75.0%** (D56, head stack), with the unambiguous subset at 91.7%
and every remaining error attributable to a named phenomenon rather than to noise.

The pattern across D52-D56 is consistent: each mechanism that matched a specific linguistic
dependency (features, position, heads) moved the score, and each remaining error needs its own
mechanism. There is no single fix left, which means the honest next step is to pick one phenomenon
and build the cue for it — `determiner_noun` (override) is the smallest and most self-contained.

## D57 · Two bugs in the evaluation harness, not in the model

Before adding any new mechanism, the harness itself was audited. Both bugs inflated the apparent
failure rate, and one of them was hiding a 100% result.

### Bug 1: unscorable pairs were counted in the denominator

`bucket(pr.phenomenon).n++` ran for every pair, including the ones skipped because a word in the
context or a candidate was unknown to the model. `ok` only ever counted scorable pairs, so every
per-phenomenon accuracy was understated by an amount that depended on how many words happened to be
in the vocabulary. Fixed by moving the increment after the skip check.

### Bug 2: four of the six `determiner_noun` pairs were labelled backwards

The pair table marked the *ungrammatical* continuation as the correct one:

| context | table said "good" | actually grammatical |
|---|---|---|
| this | car | car |
| these | car | **cars** |
| that | house | house |
| those | house | **houses** |
| many | water | **waters** |
| much | waters | **water** |

The model was choosing the grammatical noun and being scored wrong for it. Correcting the four
inverted labels moved `determiner_noun` from 2/4 to **4/4 = 100%** with no change to the model at
all.

### Corrected table

Same configuration as D56 (3M tokens, 6 slots, hybrid feature code, xi=1e-4, head stack on,
head_w=4):

| phenomenon | n | correct | acc |
|---|---|---|---|
| agreement_adjacent | 6 | 6 | 100.0% |
| agreement_long | 6 | 5 | 83.3% |
| determiner_noun | 4 | 4 | **100.0%** |
| aux_agreement | 2 | 2 | 100.0% |
| one_of | 2 | 0 | 0.0% |
| **unambiguous** | 12 | 11 | 91.7% |
| **ambiguous** | 8 | 6 | 75.0% |
| **OVERALL** | 20 | 17 | **85.0%** |

Forced-choice agreement across the whole line of work: 57.1% (D51) -> 60.7% (D52) -> 70.0% (D54)
-> 75.0% (D56) -> **85.0%**, of which 5 points came from fixing the harness rather than the model.

### What is left

One phenomenon fails outright and one pair fails:

1. `one_of` (0/2): "one of the keys" must take a singular verb. The correct head is the quantifier
   `one`, not the plural noun `keys`. Pushing `one` onto the head stack was tried and does not help:
   the head matrix for `one` receives ~123k updates overall but the rows for `one` stay at exactly
   0.0, which means the update path for that head is not being reached the way the relativiser path
   is. This is now a code-level question, not a modelling one.
2. One `agreement_long` pair: "the man who the women saw" -> `left`. The embedded subject `women` is
   plural and nearer than the head `man`, so the model follows the nearer noun. Agreement attraction
   one level down; the head stack picks the right head but the distance slots still outvote it.

Both are head-identification problems, and both are narrow enough to be settled by direct
inspection rather than by another sweep.

### Harness changes made

- `head_mode` / `head_w`: the head stack (relativiser and quantifier heads).
- `ctx_mode=3`: truncate the context at a relativiser, for the access experiment.
- `GRAMMAR_BLOCKS`: per-slot and aggregate feature-block weights.
- `GRAMMAR_FEATONLY`: score using feature bits only.
- `feat_boost`, `freq_floor`, `readout_mode`, `readout_temp`, `pos_w`: the knobs used in D52-D55.
- `gen_agree.py` now generates from the same lexicon as the pairs, including past-tense relative
  verbs and `men`/`women`, so the agreement-at-a-distance pairs are scorable at all.

## D58 · A featureless word had an empty code, so it could not be a head

`one_of` failed at 0/2 and the head rows for `one` were exactly 0.0. The cause was in the code
builder, not the learning rule: in `code_mode>=4` the identity bits were only added to words that
had a feature-table entry, so a word with no feature got **zero bits**. `one` is a quantifier with
no feature, so its code was empty: invisible to the readout, and unable to receive a head update
because `w_head` is indexed by the head's bits. 33,334 head updates were being computed for `one`
and written to nothing.

The first attempt gave the fallback only to featureless words. That fixed `one_of` (0/2 -> 2/2)
but broke `agreement_adjacent` (100% -> 50%), because every function word suddenly carried four
random bits and diluted the feature signal. Giving **every** word four identity bits plus its
features kept the quantifier visible without the dilution: four identity bits against sixteen
feature bits is a 20% identity share, where the old twelve-bit identity code was a third.

Result: `one_of` 0/2 -> **2/2**, and `agreement_adjacent` back at 100%.

## D59 · Agreement attraction needed three structural fixes, not a weight

Four pairs failed and all four were the same error: the model agreed with the nearest noun.

| context | model said | correct | nearest noun |
|---|---|---|---|
| the keys on the table | turns | turn | table (sg) |
| the boxes in the room | is | are | room (sg) |
| the box that the children opened | were | was | children (pl) |
| the man who the women saw | leaves | left | women (pl) |

Three changes, all structural:

1. **A preposition opens a modifier phrase.** The noun after `on`/`in`/`with`/`near` is inside that
   phrase and is not the subject. Without this, `table` became the head of "the keys on the table".
2. **The head falls back to the last noun outside a modifier phrase.** The stack was only pushed by
   relativisers and quantifiers, so an ordinary subject had no head at all and the distance slots
   decided. The head is now `head_stack.front()` if the stack is non-empty, else `last_noun`.
3. **The slots are down-weighted once a head exists.** They are the source of the error, so they
   still vote but at `slot_head_w` (default 0.25). Sweeping `slot_head_w` from 0.0 to 1.0 leaves
   the score at 100%, so the gain comes from 1 and 2, not from the weight.

Result: 85.0% -> **95.0%**; `agreement_adjacent` 6/6, `agreement_long` 5/6.

## D60 · The 100% was overfitting, and the held-out test found the real bug

After D59 the tuned pairs scored 100% at every `xi` from 1e-5 to 1e-3. That is the profile of a
memorised table, so a held-out set was written from the same lexicon but different nouns, verbs and
prepositions (`data/agree_heldout.txt`). Note that `agree_train.txt` is 335k tokens; the `max_tokens`
argument was set higher than the corpus, so every run used all of it.

Held-out scored **57.9%**, and every failure had the same shape: the model preferred the *-s* form.
`the files near the room` -> `works`, `the waters in the park` -> `finds`, `the coach` -> `coaches`.

The cause was the feature table. `works`, `opens` and `finds` were not listed, so they fell through
to the morphological rule *ends in -s -> plural* and were labelled `F_PLURAL`. A plural head
therefore predicted them. The tuned pairs used `turn`/`turns`, which were listed, so the bug was
invisible. The same gap hit `coach`, `room`, `table`, `park`, `woman` (no feature at all) and
`women` (no -s, so no plural feature).

Fix: the table now covers the whole lexicon, and every 3sg form is listed explicitly so the -s rule
cannot reach it.

Held-out: 57.9% -> 84.2% (verbs) -> **100%** (nouns). Tuned stayed at 100%.

**The lesson is the measurement, not the mechanism.** A single test set cannot distinguish a rule
from a lookup. The held-out set is now part of the harness and is what the number should be quoted
from.

## D61 · Nested relative clauses need the outermost head

The hard set (`data/agree_hard.txt`: two stacked PPs, nested relatives) scored 94.7%, with one
failure: `the coach that the women who the men saw found` and its plural twin both resolved to
`head=41`, the embedded `women`. The stack was read with `back()`, which returns the innermost
clause's subject, but the main verb agrees with the outermost subject. Reading `front()` fixes it.

Hard set: 94.7% -> **100%**. All three pair sets together: 58/58.

## D62 · The mechanism works on clean data, not yet on real text

The same configuration on wikitext-103:

| corpus | tokens | overall |
|---|---|---|
| synthetic (`agree_train`) | 335k | 100.0% (20/20) |
| synthetic held-out | 335k | 100.0% (19/19) |
| synthetic hard | 335k | 100.0% (19/19) |
| wikitext | 3M | 64.3% (18/28) |
| wikitext | 20M | **57.1%** (16/28) |

Real text does not improve with data: 3M gives 64.3%, 20M gives 57.1%. On a clean corpus the only
continuation of a subject is the agreeing verb, so the head matrix has one thing to learn. In
wikitext the same words appear in thousands of unrelated contexts and the agreement signal is a
small fraction of the association, so more data mostly adds noise.

**A note on the synthetic token counts.** `data/agree_train.txt` is 335k tokens (60k sentences of
about seven words). Every synthetic run in D50-D62 consumes all of it. Where an earlier entry says
"1.5M tokens" or "3M tokens" for the lexicon corpus, that was the `max_tokens` cap on the command
line, not the number of tokens actually read. The percentages in those entries are unaffected; only
the label was wrong. The wikitext numbers (3M, 20M) are real, because that corpus is large enough
to reach the cap.

This is the honest state of the grammar line: the architecture can learn a feature-level agreement
rule when the data isolates it, and it cannot yet extract that rule from natural text. The next
step is not another head mechanism but a way to make the agreement signal separable in noisy input
(for example, learning which contexts are clause-internal before associating).

### Harness changes made in D58-D62

- Every word in `code_mode>=4` gets four identity bits plus its features (D58).
- `in_pp` tracks modifier phrases; `head_bits()` falls back to `last_noun` and reads `front()` (D59,
  D61).
- `slot_head_w` (argv[21]) scales the distance slots when a head is known (D59).
- A pair file can be loaded from argv[22] in the form `phenomenon|context|good|bad|amb`, so held-out
  sets are scored by the same binary (D60).
- `data/agree_heldout.txt` and `data/agree_hard.txt` are new held-out sets (D60, D61).
- The feature table now covers the whole lexicon (D60).
