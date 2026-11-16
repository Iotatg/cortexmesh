# 02 · Biology → computation map

One row per biological mechanism. "Primitive" = the exact thing we implement.

| # | Biology | Primitive | Where it lives in code |
|---|---|---|---|
| 1 | Sparse distributed code (~2 %) | `SDR`: fixed-size bit vector, k active | `bm/sdr.hpp` |
| 2 | Miniature column of ~100 neurons | `Column`: 1 SDR column = 100 cells | `bm/column.hpp` |
| 3 | Feedforward drive → L4 | proximal dendrite: `k` synapses, overlap → activation | `bm/spatial_pooler.hpp` |
| 4 | Lateral prediction (L2/3) | distal segments on cells; active-segment → predictive state | `bm/temporal_memory.hpp` |
| 5 | Apical feedback (L1) | `feedback` bus: top-down SDR added as bias to columns | `bm/region.cpp` |
| 6 | Dendritic branch nonlinearity | each segment = independent branch, saturating | `bm/temporal_memory.hpp` |
| 7 | Hebb + STDP | eligibility `e = pre_trace * post_trace * sign(dt)` | `bm/plasticity.hpp` |
| 8 | BCM sliding threshold | per-cell `theta_m` tracks mean post activity | `bm/plasticity.hpp` |
| 9 | Homeostatic scaling | per-cell input weight normalization | `bm/plasticity.hpp` |
| 10 | Three-factor rule | `Δw = η · e · m` with broadcast `m` | `bm/modulator.hpp` |
| 11 | Lateral inhibition (WTA) | k-WTA: top-k columns win, others suppressed | `bm/spatial_pooler.hpp` |
| 12 | Thalamic gating | `Gate`: scalar/row mask on input stream | `bm/gate.hpp` |
| 13 | Cortico-thalamic loop | region→gate→region feedback path (delayed) | `bm/loop.cpp` |
| 14 | Theta/gamma rhythm | global tick: 1 gamma cycle = 1 discrete step | `bm/clock.hpp` |
| 15 | Phase coding | phase of activation inside a cycle = order tag | `bm/phase.hpp` |
| 16 | Place cells | position encoder (2D) | `bm/spatial.cpp` |
| 17 | Grid cells (multi-module, 3:1 spacing) | `GridModule`: periodic phase code, multi-scale | `bm/spatial.cpp` |
| 18 | Head-direction cells | orientation ring attractor | `bm/spatial.cpp` |
| 19 | Path integration | velocity → position integration with noise | `bm/spatial.cpp` |
| 20 | Dentate gyrus (pattern separation) | high-sparsity random expansion + strong inhibition | `bm/pattern_sep.hpp` |
| 21 | CA3 attractor (pattern completion) | recurrent associative memory, WTA settling | `bm/attractor.hpp` |
| 22 | CA1 mismatch | `prediction_error = input_sdr XOR predicted_sdr` | `bm/region.cpp` |
| 23 | Sharp-wave ripple / replay | offline compressed replay of recent sequences | `bm/replay.cpp` |
| 24 | Sleep consolidation | replay → slow cortical plasticity (high η, low noise) | `bm/consolidate.cpp` |
| 25 | Complementary learning systems | two learning rates: fast (hippocampus) / slow (cortex) | `bm/region.hpp` |
| 26 | Basal ganglia action selection | candidate actions → WTA → gate; D1/D2 as two thresholds | `bm/basal_ganglia.hpp` |
| 27 | Dopamine RPE | `rpe = reward - predicted_reward`; EMA predictor | `bm/modulator.hpp` |
| 28 | Eligibility trace | decaying `e` per synapse (~1 s) | `bm/plasticity.hpp` |
| 29 | Cerebellum: expansion recoding | random sparse expansion (granule layer) | `bm/cerebellum.hpp` |
| 30 | Cerebellum: climbing-fiber error | supervised delta on Purkinje output | `bm/cerebellum.hpp` |
| 31 | Cerebellum: forward model | predicts sensory consequence of motor command | `bm/cerebellum.hpp` |
| 32 | Persistent activity (PFC) | recurrent attractor with decay, gated by DA | `bm/working_memory.hpp` |
| 33 | Attention = gain | per-region/per-column gain scalar (ACh/NA proxy) | `bm/modulator.hpp` |
| 34 | Neuromodulator broadcast | `Modulators { rpe, novelty, ach, na }` struct, global | `bm/modulator.hpp` |
| 35 | Synaptic tagging | mark potentiated synapses for later consolidation | `bm/plasticity.hpp` |
| 36 | Structural growth | add/remove synapses when a segment is underused | `bm/temporal_memory.hpp` |
| 37 | Inhibition (PV/SOM/VIP) | per-column inhibition + branch gating | `bm/interneuron.hpp` |
| 38 | Hierarchy of areas | `Region` graph; each region predicts its input | `bm/region.hpp` |
| 39 | Temporal pooling (stable objects) | pool over last N steps, gated by novelty | `bm/pooler.cpp` |
| 40 | Embodiment / action | `motor` output = action SDR; effect appended to input | `bm/agent.cpp` |

## Rules that follow

1. **No dense matrices as state.** A region holds sparse synapses, not a weight matrix.
   Dense only as a temporary buffer for computing overlaps, and it is discarded.
2. **No backward pass.** Every update is computed from local traces. The only global
   signal is a small set of scalars.
3. **Delay is a first-class parameter.** Each connection has `delay_ticks`; predictive
   segments fire `delay` steps after their cause.
4. **Everything is a stream of SDRs.** Input, output, prediction, reward: all SDRs on a
   shared tick.
5. **Consolidation is separate from learning.** Online = fast, local, noisy.
   Offline = slow, repeated, compressed.