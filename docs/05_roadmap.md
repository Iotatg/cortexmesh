# 05 · Roadmap and the 10,000-question research backlog

## Guiding rule

Every phase must end with a **measurable** result, not "it feels smarter". Metrics are in
`11_benchmarks.md` (to be written with the first working benchmark).

## Phase 0 — Foundations (now)

Deliverables:
1. `docs/01..04` (done).
2. `libbrain` core primitives: SDR, encoder, spatial pooler, temporal memory.
3. Tests that prove: sparse codes overlap correctly; SP learns stable representations;
   TM learns sequences and predicts; TM flags anomalies.
4. A CLI to run them on toy data.

Exit gate: `ctest` green; TM predicts next element of a random deterministic sequence with
> 95 % accuracy after one pass.

## Phase 1 — Single region, unsupervised

- Add delays, phase, temporal pooling.
- Benchmark: sequence prediction on synthetic grammars (a^n b^n, nested, Markov).
- Add anomaly detection; measure ROC on corrupted inputs.
- Add homeostatic control; verify no dead columns after 10^6 ticks.

## Phase 1.5 — Modulators, salience, focus, interoception

- `ModulatorSystem`: dopamine RPE, noradrenaline novelty, acetylcholine precision.
- `Salience`: anomaly/uncertainty → gain and effective learning rate.
- `Interoception` input channel (body state → SDR, biases value and precision).
- `Region::set_focus`: learning gain + noise (inverted-U).
- Test: moderate focus learns faster than extreme focus.

## Phase 2 — Hierarchy + feedback

- Two/three regions; each predicts its input.
- Benchmark: learn a hierarchical grammar (e.g., digits → numbers → expressions).
- Show that lower-level prediction error falls as higher level learns.

## Phase 3 — Hippocampal loop

- Dentate expansion, CA3 attractor, CA1 mismatch.
- One-shot learning of arbitrary associations; replay; consolidation into cortex.
- Benchmark: paired-associate learning (A:B, B:C) and inference (A:C).

## Phase 4 — Basal ganglia + action

- Reward-driven action selection; RPE; eligibility traces.
- Benchmark: grid-world navigation, then a simple bandit with delayed reward.

## Phase 5 — Cerebellum + motor

- Supervised error correction, forward model.
- Benchmark: predict sensory consequence of action; inverse model control.

## Phase 6 — Spatial system

- Grid/place/head-direction; path integration.
- Benchmark: navigate a 2D maze with dead reckoning, then landmark correction.

## Phase 7 — Language / symbols

- Token encoder, sequence memory, hierarchical syntax.
- Benchmark: next-token prediction on a small corpus; compare to n-gram and tiny
  transformer baselines (for reference only — not a target).

## Phase 8 — Consolidation and sleep

- Offline replay; measure catastrophic forgetting with/without replay.
- Benchmark: sequential task learning (A then B), retention of A.

## Phase 9 — Integration

- Agent loop: sense → predict → act → reward → consolidate.
- Benchmark: a small suite of tasks with a fixed compute budget.

## The 10,000 questions: taxonomy and process

We do not need 10,000 *answers* before writing code; we need a backlog where every
implementation decision can point to a question it resolves. Structure:

### A. Biology (target ~2,000)
A1 cellular (channels, dendritic computation, synapse types, glia)
A2 microcircuit (layers, interneuron classes, connectivity statistics)
A3 systems (sensory, motor, memory, valuation)
A4 development (critical periods, myelination, pruning)
A5 pathology as reverse-engineering (epilepsy, Parkinson's, amnesia, autism models)

### B. Coding theory (target ~1,200)
B1 sparsity vs. capacity; B2 overlap as similarity; B3 union/intersection algebra;
B4 temporal codes; B5 phase coding; B6 population geometry; B7 noise robustness;
B8 hash collisions; B9 compositional codes; B10 binding problem.

### C. Learning (target ~1,500)
C1 STDP variants; C2 BCM; C3 homeostatic scaling; C4 eligibility traces;
C5 credit assignment over delays; C6 structural plasticity; C7 stability-plasticity;
C8 catastrophic forgetting; C9 meta-learning (learning rate as a state).

### D. Memory (target ~1,500)
D1 working memory capacity and gating; D2 episodic binding; D3 consolidation;
D4 replay ordering; D5 interference; D6 schema formation; D7 forgetting curves.

### E. Control (target ~1,200)
E1 action selection; E2 exploration; E3 model-based vs. model-free; E4 hierarchical RL;
E5 timing; E6 error correction; E7 forward models.

### F. Perception and world models (target ~1,000)
F1 predictive coding; F2 object permanence; F3 spatial maps; F4 sensor fusion;
F5 active inference; F6 abstraction.

### G. Language (target ~1,000)
G1 tokenization as sparse codes; G2 syntax as sequence memory; G3 semantics as
conjunctions; G4 reference/grounding; G5 compositionality; G6 long-range dependencies
without attention.

### H. Computation/systems (target ~1,000)
H1 memory layout for sparse data; H2 parallelism (per-column independence);
H3 numerical stability; H4 serialization/checkpointing; H5 hardware mapping (FPGA/ASIC);
H6 energy per prediction.

### I. Evaluation and safety (target ~1,000)
I1 benchmarks; I2 ablations; I3 interpretability (read the SDRs); I4 failure modes;
I5 alignment (reward hacking); I6 reproducibility.

### Process
1. Keep a single `questions.md` with ids `A1.001`, `B2.014`, ... and a status field:
   `open | answered | implemented | rejected`.
2. Every code comment that encodes a non-obvious choice cites a question id.
3. Every benchmark result cites the question it settles.
4. One review session per 100 questions; prune the backlog, do not let it rot.
5. Prefer experiments over literature: run the code, then read.

## Immediate next actions (this session)

1. Hierarchy with feedback (`Region` graph, apical bus).
2. Valuation region (vmPFC-like) + RPE wiring.
3. Dual controller: model-free (BG) vs model-based (hippocampus), arbitrated by uncertainty.
4. Interoception channel and its effect on value/precision.
5. Anomaly ROC and catastrophic-forgetting benchmarks.