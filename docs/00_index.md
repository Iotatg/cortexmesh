# 00 · Index

Vision: build an intelligent system from first principles of neuroscience, with **no
transformers, no attention, no backpropagation through time, no gradient descent on a
global loss**. Learning must be local, sparse and incremental, like cortex.

## Documents

| File | What it answers |
|---|---|
| `01_brain_research.md` | How the human brain is built and how it computes: parts, cells, coding, plasticity, neuromodulators, rhythms, sleep, energy budget. |
| `02_brain_to_computation.md` | Every biological mechanism mapped to a computational primitive we can implement. |
| `03_architecture.md` | The system we are building (`CortexMesh`): modules, dataflow, hierarchy, memory loops. |
| `04_learning_rules.md` | Exact local learning rules: Hebbian, STDP, BCM, homeostatic scaling, prediction error, dopamine RPE, eligibility traces. |
| `05_roadmap.md` | Phases, milestones, test gates, and the 10,000-question research backlog (taxonomy + process). |

## Engineering documents (once code exists)

| File | Purpose |
|---|---|
| `10_api.md` | Public API of `libbrain`. |
| `11_benchmarks.md` | What we measure and target numbers. |
| `12_emotion_logic_focus.md` | Emotion, logic, focus, body state, consciousness (research + sources). |
| `13_discoveries.md` | Measured discoveries D1–D23 and experiment results. |
| `20_phonotactics.md` | Learned word boundaries and auditory categories without labels. |
| `21_knowledge_induction.md` | Facts, multi-hop, interference, replay tradeoff. |
| `22_ordered_speech.md` | Nested theta-gamma phases, co-articulation, order tests. |
| `14_simulation_test.md
15_language_grounding.md` | Grid-world simulation: can the AI detect rule violations? |

## Non-negotiable design constraints

1. **Local learning only.** A synapse changes using information available at that synapse:
   pre activity, post activity, and a broadcast neuromodulatory scalar. No global error
   back-propagated through layers.
2. **Sparse codes.** Every representation is a sparse distributed representation (SDR),
   target sparsity 2–4 %. Dense vectors are forbidden as internal state.
3. **Prediction is the primary signal.** The network's job is to predict its next input;
   behaviour falls out of prediction plus reward gating.
4. **Online, single-pass.** One pass per sample, no epochs, no shuffling, no replay buffer
   (replay exists, but as an offline *consolidation* process, not a training batch).
5. **Fixed-size everything.** No growing tensors, no context windows, no attention over
   past states. Memory is in synapses and in persistent activity.