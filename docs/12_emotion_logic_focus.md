# 12 · Emotion, logic, body, focus, consciousness — what the research actually says

Purpose: replace the popular "emotional brain vs logical brain" picture with the current
systems view, and turn each claim into a concrete mechanism we can implement. Every claim
below is tied to a source; where the literature is contested, that is stated.

## 0. Short answer

There is **no "emotional brain" and no "logical brain" as two organs**. There are:

1. **Large-scale networks** that cooperate and compete (salience, frontoparietal control,
   default mode, dorsal attention, limbic).
2. **Valuation and control circuits** (vmPFC/OFC, amygdala, ventral striatum, ACC) that
   assign value and select actions.
3. **Neuromodulatory systems** (dopamine, noradrenaline, acetylcholine, serotonin) that
   broadcast scalar signals and set gain, plasticity and persistence.
4. **Interoceptive channels** (insula, vagus, brainstem) that report body state.

"Logic" is not a region: it is what the frontoparietal control network does when it
maintains a goal and biases competition. "Emotion" is not a region either: it is a
**valuation + prediction** process implemented across vmPFC, amygdala, insula, hypothalamus
and brainstem, which also modulates the control network.

## 1. Why the triune brain is wrong

The "reptilian / limbic / neocortex" story (MacLean, 1960s) is still popular but obsolete:
- Reptiles and birds have sophisticated pallial/cortical-like structures.
- "Limbic system" is not a single functional system; components do many jobs.
- There is no "reptilian brain" that fights the "rational brain".

Modern replacement: **network neuroscience**. The same region participates in many networks
depending on task, state and connectivity. Function is in the *interactions*, not the box.

Source: reviews of comparative neuroanatomy and network neuroscience (e.g., the
"large-scale brain network" framework; Barrett's constructed-emotion work).

## 2. The main large-scale networks

| Network | Core nodes | Job | Failure signature |
|---|---|---|---|
| Salience network (SN) | anterior insula, dorsal ACC | detect behaviourally relevant events, switch between DMN and FPN | anxiety, addiction, pain |
| Frontoparietal control (FPN / central executive) | dlPFC, posterior parietal, dACC | hold goals, rules, working memory, cognitive control | ADHD, dysexecutive syndrome |
| Dorsal attention (DAN) | frontal eye fields, intraparietal sulcus | top-down spatial attention | neglect |
| Default mode (DMN) | medial PFC, PCC/precuneus, angular gyrus, hippocampus | self-reference, memory, mind-wandering, prospection | rumination, depression, hallucination models |
| Limbic / valuation | amygdala, vmPFC/OFC, ventral striatum, hypothalamus, PAG | value, salience, autonomic/endocrine output | anxiety, addiction, psychopathy |
| Interoceptive | posterior→mid→anterior insula, ACC, brainstem nuclei | body state, feeling, homeostasis | alexithymia, somatic symptom disorders |

Key modern findings:
- The **insula is heterogeneous**: it has multiple functional zones (interoceptive,
  cognitive, social), not one "emotion centre" (Cai & Menon, 2025, *Brain Struct Funct*,
  doi:10.1007/s00429-025-03027-y).
- The **salience network** is what decides which network gets control (Seeley et al., 2007;
  Menon & Uddin, 2010).
- The **DMN** is not "the task-negative network"; it is active in memory, imagination,
  social cognition and self-referential thought. Its coupling with rumination is a
  depression risk marker (Chou et al., 2023; Mısır et al., 2024).

## 3. "Logic" — what is actually active during reasoning and focus

Reasoning and working memory recruit a **fronto-parietal-cingulate** set:
- **dlPFC**: maintains rules, goals, and relational information; activity scales with
  difficulty.
- **Posterior parietal cortex**: maintains and manipulates representations.
- **dACC/pre-SMA**: conflict monitoring, effort, error detection.
- **Frontoparietal control network**: flexibly reconfigures connectivity; its
  interconnectivity dynamics predict attention control and cognitive performance
  (Seeburger et al., 2026, *PNAS*, doi:10.1073/pnas.2526828123).

**Focus is not "logic switching on".** It is a **gain change**:
- **LC–NE system** sets the gain of cortical processing; tonic vs phasic modes trade
  exploration for exploitation (Aston-Jones & Cohen, 2005). LC-NE integrity predicts fluid
  intelligence and attention control (Tsukahara & Engle, 2021, *PNAS*,
  doi:10.1073/pnas.2110630118).
- **Cholinergic** input from basal forebrain increases the precision of sensory evidence and
  cortical plasticity.
- **Noradrenergic** rhythmic activity shapes selective attention (van den Brink et al.,
  2022, *Trends Cogn Sci*).
- Focus also **suppresses the DMN**; failure to suppress is linked to mind-wandering and
  lapses.

So when you "focus", you do not activate a logic module; you raise the gain of the current
task network, suppress competing networks, and increase the precision of prediction errors.
The feeling of "clear thinking" is the subjective side of that gain change.

**Caveat:** the popular "ego depletion"/"willpower as fuel" story is contested; replication
is weak. Effort and fatigue are real, but the mechanism is not a simple fuel gauge.

## 4. "Emotion" — what is actually happening

Emotion is best modelled as **valuation + prediction + interoceptive feedback**, not as a
separate primitive system.

- **Amygdala**: rapid relevance/salience detection (especially threat and reward cues);
  it does not "store" emotions. It modulates sensory processing and memory consolidation.
- **vmPFC/OFC**: computes subjective value, compares options, updates value on feedback.
- **Ventral striatum**: reward prediction error (dopamine) and incentive salience.
- **Anterior insula**: interoceptive prediction error — "how does my body feel relative to
  what I expected?"
- **Hypothalamus/PAG**: autonomic and behavioural output (fight/flight, freezing,
  approach).
- **ACC**: effort, conflict, and prediction error at the action level.

Two useful models:

1. **Somatic marker hypothesis** (Damasio, 1994): bodily states are used as fast priors in
   decision-making. Supported in part, but the strong version (body state *causes* the
   decision) is contested; interoceptive signals are *one input*, not the decider.
2. **Theory of constructed emotion** (Barrett): emotions are not natural kinds with fixed
   circuits; the brain constructs an emotional episode from core affect (valence/arousal),
   interoception, concepts and context. Evidence: high variability of neural patterns for
   the same emotion category; context dependence. Jungilligens et al. (2022, *Brain*) use
   this to reframe functional neurological disorder.

**Practical consequence for us:** implement emotion as **two scalars plus a bias**:
valence (approach/avoid), arousal (gain), and a value estimate per state-action. Do not
build an "emotion module" with a lookup table.

## 5. Body, heart and gut — interoception

- **Interoception** = sensing internal state: heart rate, breathing, gut, temperature,
  inflammation. The insula is the main cortical hub; the vagus nerve is the main
  parasympathetic channel; brainstem nuclei (NTS, parabrachial) relay.
- Heartbeat-evoked potentials and heart-rate variability (HRV) correlate with decision-making,
  emotion regulation and risk-taking. HRV is a proxy for prefrontal-autonomic regulation.
- **Vagus nerve**: bidirectional signalling; recent work maps its sensory processing
  (e.g., "What happens in vagus", 2026, *Front Neurol*).
- **Gut–brain axis**: microbiome, vagal afferents, immune signalling; effects on mood and
  cognition are real but mostly small and context-dependent.

**Where your model may be wrong:** the heart is not "the emotional organ". It is a *sensor*
whose signals are interpreted by the insula and used as priors. "Heart brain" (40k neurons
in the heart) is real anatomy, but it is a local cardiac plexus, not a second brain.

## 6. Consciousness

Still unsolved. Two main theories:

- **Global Neuronal Workspace (GNW)**: conscious content is broadcast globally; prefrontal-
  parietal ignition. (Mashour, Roelfsema, Changeux, Dehaene, 2020, *Neuron*.)
- **Integrated Information Theory (IIT)**: consciousness = integrated information (Φ);
  posterior "hot zone".

The **Cogitate adversarial collaboration** (2025, *Nature*, doi:10.1038/s41586-025-08888-1)
tested both with intracranial recordings and found **neither theory fully predicted the
data** — evidence for both, problems for both. Anesthesia studies (Mashour, 2024,
*Neuron*) show that consciousness depends on **network integration and thalamocortical
loops**, not a single "consciousness centre".

Implication for us: do not build a "consciousness module". Build **global broadcast,
integration, and thalamocortical-like loops**, and measure integration.

## 7. Dual-process theory (System 1 / System 2) — useful but overstated

- Kahneman's System 1/2 is a useful shorthand, but the "two systems" claim is contested
  (Melnikoff & Bargh, 2018, *Trends Cogn Sci*: "the mythical number two"). Evidence supports
  **multiple interacting processes**, not two clean systems.
- What survives: a fast, model-free, habitual controller (basal ganglia, cerebellum) and a
  slow, model-based, goal-directed controller (prefrontal–hippocampal). They compete and
  are arbitrated by uncertainty, reward rate and cognitive cost.

**Implementation:** two controllers + an arbitrator, not two "brains".

## 8. What this means for `CortexMesh` (concrete design)

We add **six things**, all local, all scalar-or-sparse:

1. **Modulators** (`bm/modulator.hpp`):
   - `rpe` = reward prediction error (dopamine proxy).
   - `novelty` = mismatch between prediction and input (noradrenaline proxy).
   - `ach` = precision/focus gain (cholinergic proxy).
   - `na` = reset/gain spike on surprise.
   All are scalars broadcast to every synapse; the three-factor rule uses them.

2. **Salience** (`bm/salience.hpp`): computes anomaly and precision, sets `gain` and
   `learning_rate`. High anomaly → high gain, high plasticity (novelty), then decay.

3. **Interoception channel**: a dedicated input SDR (body state: heart rate, HRV, breathing,
   "effort"). It does not drive actions directly; it biases value and precision. This is the
   computational version of the insula loop.

4. **Valuation region** (`vmPFC`-like): maps state SDR → value SDR; learns by RPE-modulated
   Hebb; supplies `predicted_reward` to the modulator system.

5. **Two controllers + arbitrator**:
   - model-free: basal-ganglia-style WTA over cached action values;
   - model-based: hippocampus/cerebellum forward model, one-step lookahead;
   - arbitrate by uncertainty × cost (this is "willpower" without a fuel tank).

6. **Focus as gain, not module**: `ach` scales prediction error in every region and
   suppresses the DMN-like self-model region (lower its gain), reproducing attentional
   selection.

### Minimal formulas

```
rpe_t     = r_t + gamma * V(s_{t+1}) - V(s_t)
V(s)     += alpha_v * rpe_t                      (per state SDR)
novelty_t = (1 - lambda) * novelty_{t-1} + lambda * anomaly_t
ach_t     = base + k * uncertainty_t             (uncertainty = running variance of anomaly)
na_t      = max(0, novelty_t - theta_na) * decay
gain      = ach_t * (1 + beta * na_t)
lr_eff    = lr_base * gain
```

### What we will *not* do

- No "emotion module" with a lookup table.
- No "consciousness box".
- No claim that focus = logic activation; focus = gain + competition.
- No triune brain.

## 9. Where your original assumptions were right and wrong

| Assumption | Verdict |
|---|---|
| "Alag alag parts, alag kaam" | Right at the level of *networks and neuromodulators*, wrong at the level of "one box per function" |
| "Focus karne pe logic part active hota hai" | Half right: focus raises gain of the *current task network* (often frontoparietal), and suppresses DMN; it is not a logic switch |
| "Emotional brain vs logical brain" | Wrong as two organs; better: valuation network vs control network, coupled |
| "Heart = emotion" | Wrong: heart is an interoceptive sensor; insula interprets it |
| "Consciousness ek part hai" | Wrong: it is a network-level process (integration + broadcast), still unexplained |

## 10. Sources (with year, venue, identifier)

Recent (2019–2026):
1. Cai W, Menon V. *Heterogeneity of human insular cortex: five principles…* Brain Struct
   Funct, 2025. doi:10.1007/s00429-025-03027-y
2. Seeburger DT et al. *Attention control ability is associated with frontoparietal control
   network interactions.* PNAS, 2026. doi:10.1073/pnas.2526828123
3. Tsukahara JS, Engle RW. *Fluid intelligence and the locus coeruleus–norepinephrine
   system.* PNAS, 2021. doi:10.1073/pnas.2110630118
4. van den Brink RL et al. *Noradrenergic modulation of rhythmic neural activity shapes
   selective attention.* Trends Cogn Sci, 2022.
5. Woo E, Sansing LH, Arnsten AFT, Datta D. *Chronic stress weakens connectivity in the
   prefrontal cortex.* Chronic Stress, 2021. doi:10.1177/24705470211029254
6. Mashour GA. *Anesthesia and the neurobiology of consciousness.* Neuron, 2024.
   doi:10.1016/j.neuron.2024.03.002
7. Cogitate Consortium et al. *Adversarial testing of global neuronal workspace and
   integrated information theories of consciousness.* Nature, 2025.
   doi:10.1038/s41586-025-08888-1
8. Mashour GA, Roelfsema P, Changeux JP, Dehaene S. *Conscious processing and the global
   neuronal workspace hypothesis.* Neuron, 2020. doi:10.1016/j.neuron.2020.01.026
9. Berboth S, Morawetz C. *Amygdala–prefrontal connectivity during emotion regulation: a
   meta-analysis.* Neuropsychologia, 2021.
10. Jungilligens J et al. *A new science of emotion: implications for functional
    neurological disorder.* Brain, 2022. doi:10.1093/brain/awac204
11. Chou T et al. *The default mode network and rumination in individuals at risk for
    depression.* SCAN, 2023.
12. Mısır E et al. *Functional connectivity in rumination: systematic review.* J Clin Exp
    Neuropsychol, 2024.
13. Evans AK et al. *Impact of noradrenergic inhibition on neuroinflammation…* J
    Neuroinflammation, 2024.
14. *What happens in vagus.* Front Neurol, 2026.

Classics (still load-bearing):
15. Damasio AR. *Descartes' Error* (1994) — somatic marker hypothesis.
16. LeDoux J. *The Emotional Brain* (1996) — amygdala and threat.
17. O'Keefe & Nadel (1978); Hafting et al. (2005) — place/grid cells.
18. Schultz, Dayan, Montague (1997) — dopamine RPE.
19. Miller & Cohen (2001) — anterior cingulate and prefrontal control.
20. Seeley et al. (2007) — salience network.
21. Raichle et al. (2001) — default mode network.
22. Aston-Jones & Cohen (2005) — LC-NE adaptive gain.
23. Melnikoff & Bargh (2018) — "the mythical number two" (dual-process critique).
24. Mountcastle (1997) — cortical column as the unit of computation.
25. Hawkins & Ahmad (2016); Hawkins et al. (2024/2025) — HTM and Thousand Brains.

Very recent preprints (2025–2026, non-peer-reviewed):
26. *Local learning is all you need: brain-aligned predictive coding networks.* arXiv,
    2026-10.
27. *The Bayesian Reflex: a predictive coding engine for AI.* arXiv, 2026-08.
28. *The Thousand Brains Theory 2.0.* arXiv, 2025-07.
29. *Learn from your own latents and not from tokens: a sample-complexity theory.* arXiv,
    2026-05.
30. *Neuro-Vesicles: neuromodulation should be a dynamical system.* arXiv, 2025-12.
31. *Distributed Hierarchical Temporal Memory…* arXiv, 2026-06.

## 11. Open questions (to add to `questions.md`)

| id | question |
|---|---|
| J1.001 | Does a scalar `ach` gain reproduce attentional selection in our regions? |
| J2.001 | Can valence be stored as extra SDR bits without breaking overlap semantics? |
| J3.001 | Does interoceptive input improve value learning under delayed reward? |
| J4.001 | What arbitrates model-free vs model-based control in our system? |
| J5.001 | Can global broadcast (GW-like) be implemented as an SDR OR over regions? |
| J6.001 | How to measure "integration" (IIT-like) in a sparse network? |
| J7.001 | Does high arousal (high gain) help or hurt prediction accuracy? (expect inverted-U) |
| J8.001 | Can we induce a "mind-wandering" state by lowering `ach` and raising DMN gain? |