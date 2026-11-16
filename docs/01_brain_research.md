# 01 · How the brain works (research digest)

Scope: only facts that constrain an implementation. Numbers are order-of-magnitude and
flagged `[verify]` where the literature disagrees.

## 1. Scale and budget

- ~86 billion neurons; ~85 billion non-neuronal glia. `[verify: Herculano-Houzel 2009]`
- ~16 billion cortical neurons, ~69 billion cerebellar neurons (cerebellum is 10 % of mass
  but > 50 % of neurons). `[verify]`
- ~10^14–10^15 synapses. Average cortical neuron: ~7,000–10,000 synapses.
- Brain power ~20 W; a single cortical column operation costs ~10^-15 J.
  Consequence: **energy is the real constraint, not FLOPs**. Sparsity is not an
  optimization, it is the point.
- Conduction delays 1–20 ms; cortical loop (thalamus→cortex→thalamus) ~10–30 ms;
  a percept-to-action cycle ~100–200 ms.
  Consequence: **time constants matter**. Any model that ignores axonal delay cannot
  produce oscillations, phase coding or sequence learning.

## 2. Cellular building blocks

**Neuron.** Soma, dendrites (input), axon (output), synapses (contact points). Membrane is
a leaky integrator: `C dV/dt = -g_L (V - E_L) + I_syn`. Spike when `V` crosses threshold,
then refractory (~1–2 ms absolute, ~5–10 ms relative). This is a *nonlinearity that costs
nothing* — no activation function, no bias term, just a threshold on an integrator.

**Dendrites are not passive cables.** Each dendritic branch supports local NMDA spikes
(~10–50 per branch, ~20–100 µm, ~5–20 ms). A neuron therefore computes
`spike = f(sum over branches of branch_nonlinearity(local_input))`. That is a
two-layer network per neuron, learned locally, with no backprop.
This is the single most important biological fact for architecture: **depth exists inside a
single cell**.

**Synapse types.** Excitatory (glutamate, AMPA fast ~1 ms, NMDA slow ~50–100 ms and
voltage-dependent → coincidence detector), inhibitory (GABA_A fast, GABA_B slow).
Dendrite-targeting interneurons (SOM) gate branch plasticity; VIP interneurons gate SOM.

**Neuromodulators (broadcast scalars, diffuse volume transmission).**
| System | Signal | Effect on learning/computation |
|---|---|---|
| Dopamine (VTA/SNc) | reward prediction error | gates plasticity, tags eligibility, sets exploration |
| Acetylcholine (basal forebrain) | uncertainty / attention | increases plasticity, sharpens cortical representations |
| Noradrenaline (LC) | novelty / urgency | resets, increases gain, triggers network reset |
| Serotonin (raphe) | patience / harm aversion | discounts future reward, slows learning |
| Histamine, orexin | arousal / wakefulness | global gain |
Key property: these are **scalar broadcasts**, not per-synapse error signals. So credit
assignment must come from local eligibility traces, not from a backward pass.

## 3. Cortical microcircuit (the canonical column)

Six layers, ~80 % of cortical neurons excitatory pyramidal, ~20 % inhibitory.
- **L4** thalamic input (feedforward drive).
- **L2/3** intracortical, projects to other columns/areas (feedforward output).
- **L5** deep, projects to subcortical targets, also apical dendrites in L1 (feedback).
- **L6** cortico-thalamic, modulates thalamic gain.
- **L1** receives top-down feedback and long-range context, mostly modulatory (dendritic
  input, does not drive spiking by itself).
Minicolumn: ~80–100 neurons sharing a receptive-field preference, spanning all layers.
Hypercolumn: ~50–100 minicolumns covering a full feature set for a patch of input.

Three input streams, kept separate on purpose:
1. **Feedforward (bottom-up)** → L4 → L2/3: drives cells.
2. **Lateral (within column / within area)** → L2/3: predicts, disambiguates.
3. **Feedback (top-down)** → L1 apical dendrites: context, prediction, attention.
Prediction is computed *dendritically*; it does not directly cause a spike but it
**biases** which cells win — that is "predictive routing".

## 4. Coding: how information is represented

- **Sparse, distributed, combinatorial.** ~1–4 % of neurons active for a given stimulus.
  Overlap between codes encodes similarity; union encodes conjunction; intersection encodes
  common features.
- **Population/rate codes** for magnitude; **temporal codes** for order and precision
  (phase precession in hippocampus: place cell fires at progressively earlier theta phase
  as the animal crosses the field).
- **Temporal sparse codes**: only a small fraction of cells are predictive at any instant.
- Capacity: with `n` neurons and `k` active, number of distinct codes is `C(n,k)`; with
  `n=2048, k=40` that is ~10^80. Sparse codes give exponential capacity in `n`.
- Noise robustness: a code survives if a fraction of bits flip, as long as overlap ranking
  is preserved.

## 5. Learning rules

**Hebb (1949).** "Cells that fire together wire together." `Δw = η · pre · post`.
- Pure Hebb is unstable (runaway). Stability comes from:
  - **LTD/LTP thresholds (BCM)**: `Δw ∝ post · (post - θ_M)` with a sliding threshold
    `θ_M` tracking mean activity → competition, selectivity, ocular dominance.
  - **Homeostatic synaptic scaling**: multiplicative rescale to keep firing rate near a
    set point.
  - **Heterosynaptic depression**: active synapses that fail to potentiate weaken
    (normalizes total weight).
- **STDP**: `Δw = A+ exp(-Δt/τ+)` if pre before post (Δt>0), `A- exp(Δt/τ-)` if post
  before pre. τ+ ≈ 17 ms, τ- ≈ 34 ms. STDP is Hebb with a causal window and is the
  substrate for sequence learning (feedforward chains).
- **Spike-timing + branch nonlinearity** → dendritic prediction of the *next* input.
- **Reward-modulated plasticity**: `Δw = eligibility(pre,post,Δt) × RPE`. Eligibility
  trace persists ~1 s, so a synapse can be credited for a reward that arrives later.

**Three-factor rule** (the one we will implement everywhere):
`Δw_ij = η · e_ij(t) · m(t)`
- `e_ij` = local eligibility (pre trace × post trace × STDP sign), decays with time
  constant τ_e.
- `m` = neuromodulatory scalar (RPE, novelty, uncertainty), broadcast globally.
- Plus normalization per neuron (total input weight conserved) → no runaway.

## 6. Memory systems (multiple, interacting)

| System | Substrate | Timescale | Computation |
|---|---|---|---|
| Sensory buffer | thalamus/cortex | 10–500 ms | transient high-fidelity code |
| Working memory | persistent activity in PFC (recurrent) | seconds | attractor state, gated by DA/BG |
| Episodic | hippocampus (CA3/CA1, dentate gyrus) | minutes–days | rapid pattern separation + binding |
| Semantic | neocortex (distributed) | days–years | slow interleaved consolidation |
| Procedural | basal ganglia, cerebellum | hours–years | reward-gated chunking, error correction |
| Working/"scratchpad" | hippocampal replay | offline (sleep, rest) | compressed replay → cortical consolidation |

Key facts:
- **Dentate gyrus** performs pattern separation (orthogonalizing similar inputs) via sparse
  mossy-fiber coding (~2–5 % active) and strong feedback inhibition.
- **CA3** is a recurrent attractor network: auto-association, pattern completion from a
  partial cue.
- **CA1** compares CA3 prediction with entorhinal input → mismatch signal → novelty.
- **Complementary Learning Systems**: hippocampus learns fast without catastrophic
  interference; neocortex learns slow with interleaving. This is exactly why a single
  network with one learning rate fails.
- **Replay**: during sharp-wave ripples (~100–200 Hz, 50–100 ms events), compressed
  sequences are replayed (forward and reverse) at ~10–20× speed. Consolidation = replay
  driving slow cortical plasticity. Sleep spindles gate cortical plasticity.
- **Systems consolidation**: memories become independent of hippocampus over days–years.

## 7. Subcortical control

**Thalamus.** Not a relay: it is a *router and gain controller*. Every cortical area talks
to thalamus and thalamus talks back (transthalamic loops). Thalamic reticular nucleus
provides global inhibition → attentional spotlight. Thalamic gating decides what enters
cortex (sensory gating during sleep, attention).

**Basal ganglia.** Action selection by disinhibition: cortex → striatum (D1 direct /
D2 indirect) → GPi/SNr → thalamus. Winner-take-all over candidate actions, with
dopamine RPE modulating corticostriatal plasticity. Implements **reinforcement learning
with eligibility traces** and action chunking (habits).

**Cerebellum.** Supervised learning: mossy fibers (context) → granule cells (expansion
recoding, ~10^11 granule cells → sparse ~1 % active) → Purkinje cells; climbing fiber
from inferior olive carries *error* (~1 Hz complex spike). Marr-Albus-Ito: parallel fiber
+ climbing fiber coincidence → LTD at parallel-fiber synapses. Function: forward model,
timing, error correction, fast supervised adaptation. Also holds an internal model of the
body/world.

**Hippocampal-entorhinal spatial system.** Grid cells (medial entorhinal, multiple modules
with different spacings, ~3:1 ratio between adjacent modules, phases offset), place cells
(hippocampus), head-direction cells (anterior thalamus), border cells, speed cells.
Path integration = velocity → position update; grid modules provide a multi-scale
periodic basis that is *uniquely decodable* and error-correcting.

## 8. Rhythms (why timing is not optional)

| Rhythm | Band | Source | Function |
|---|---|---|---|
| Gamma | 30–100 Hz | PV interneuron–pyramidal loops (PING/ING) | local feature binding, gain control |
| Beta | 13–30 Hz | cortex–basal ganglia | status quo maintenance, top-down predictions |
| Alpha | 8–12 Hz | thalamocortical | idling, inhibition of task-irrelevant cortex |
| Theta | 4–8 Hz | hippocampus, medial septum | sequence compression, phase coding |
| Sharp-wave ripple | 150–250 Hz | CA3→CA1 | replay/consolidation |
| Sleep spindle | 11–15 Hz | thalamic reticular | plasticity gating for consolidation |

Gamma is generated by E→I→E loop with ~10 ms period; PING produces a gamma burst per
input volley. This gives a **clock for free**: each gamma cycle is a discrete processing
step, and phase encodes order within a cycle (theta–gamma phase coding).

## 9. Perception = prediction (predictive coding)

 Cortical hierarchy predicts lower-level input; only the residual (prediction error) is
propagated upward. Precision (inverse variance) is encoded by **neuromodulatory gain**
(ACh/NA) — i.e., attention = gain on error units. Inference = minimizing prediction error
at every level; learning = adjusting generative weights to reduce long-run error.
Consequences for us: hierarchy is mandatory; top-down connections are as important as
bottom-up; "attention" is gain modulation, not dot-product weighting.

## 10. Development and plasticity windows

Critical periods are opened by inhibition maturation (PV interneurons + perineuronal nets)
and closed by myelination. Adult cortex is not plastic in the same way: it learns by
adding/removing synapses (structural) and by modulating gains. Lesson: **learning rate
must decay with competence, and structure must be able to grow**.

## 11. What the brain is *not*

- Not a feedforward stack. Loops everywhere (cortico-thalamo-cortical at ~10–30 ms).
- Not trained by a global loss. No backward pass, no weight transport.
- Not dense. 98 % of neurons silent at any moment.
- Not stateless. Persistent activity, short-term facilitation/depression, oscillations.
- Not sample-efficient via scale. It is sample-efficient via priors, hierarchy, embodiment,
  and sleep replay.

## 12. Open questions that will shape the build

1. How much of cortical computation reduces to "predict next sparse input at multiple
   timescales"? (HTM/predictive coding bet.)
2. What is the minimal set of neuromodulators needed for competent learning? (We bet on 3:
   RPE, novelty/uncertainty, and a global "sleep/consolidate" signal.)
3. Can hierarchical SDR regions with local Hebbian rules solve language-level tasks
   without attention? (Open; this is the biggest risk.)
4. How to get discrete symbols out of continuous dynamics? (Attractor dynamics + winner
   take-all + temporal pooling.)
5. How to make credit assignment work over long delays without BPTT? (Eligibility traces +
   replay + reward prediction error.)

## 13. Primary sources to read (reading list, not yet verified)

- Herculano-Houzel, *The Human Advantage* (2016) — neuron counts, energetics.
- Kandel, *Principles of Neural Science* (6e) — cellular/molecular canon.
- Hawkins & Ahmad, "Why neurons have thousands of synapses..." (2016); Cui et al. HTM
  theory (2017) — HTM spatial pooler/temporal memory.
- Hawkins et al., *A Thousand Brains* (2021) — grid-cell based cortical columns.
- Friston, "The free-energy principle" (2010); Rao & Ballard (1999) — predictive coding.
- O'Keefe & Nadel (1978); Hafting et al. (2005); Moser et al. — place/grid cells.
- McClelland, McNaughton & O'Reilly (1995) — complementary learning systems.
- Marr (1969); Albus (1971); Ito (2001) — cerebellar learning.
- Schultz, Dayan, Montague (1997) — dopamine RPE.
- BCM: Bienenstock, Cooper, Munro (1982); Turrigiano — homeostatic scaling.
- Bi & Poo (1998) — STDP.