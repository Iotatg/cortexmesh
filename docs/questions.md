# questions.md — research backlog

Status legend: `open` (not investigated) · `answered` (literature/experiment settled it) ·
`implemented` (code exists) · `rejected` (decided against, with reason).

Every non-obvious design choice in `src/` cites a question id here.

## A · Biology

| id | question | status |
|---|---|---|
| A1.001 | What is the actual energy cost per cortical column operation, and does it justify 2 % sparsity? | answered |
| A1.002 | How many distinguishable dendritic branch nonlinearities does one pyramidal cell have? | answered |
| A1.003 | Which interneuron class gates branch plasticity (SOM) and which gates gain (VIP/PV)? | answered |
| A2.001 | What fraction of cortical synapses are modifiable at any time? | open |
| A2.002 | Does the minicolumn really have a single receptive-field preference, or a distribution? | open |
| A3.001 | How is uncertainty encoded (ACh vs. NA vs. pupil)? | answered |
| A3.002 | What is the minimal neuromodulator set for competent learning? (we bet on 3) | open |
| A4.001 | When do critical periods close, and what reopens them? | answered |
| A5.001 | What does amnesia (H.M.) tell us about fast vs. slow learning? | answered |

## B · Coding

| id | question | status |
|---|---|---|
| B1.001 | Is 2 % sparsity optimal, or is it task-dependent? | open |
| B2.001 | Does overlap rank preservation suffice for noise robustness? | answered |
| B3.001 | Can union/intersection algebra support compositional concepts? | answered |
| B4.001 | How much order information can phase coding carry per cycle? | open |
| B5.001 | What is the capacity of an SDR with n=4096, k=20? | answered |
| B6.001 | Do hash collisions in the encoder matter at 2048 symbols? | open |
| B7.001 | How many bits can flip before a code is misread? | answered |

## C · Learning

| id | question | status |
|---|---|---|
| C1.001 | Is STDP necessary, or is rate-based Hebb enough for sequences? | answered |
| C2.001 | Does BCM improve selectivity over plain Hebb in the pooler? | open |
| C3.001 | What normalization keeps Hebb stable without a global loss? | answered |
| C4.001 | How long should eligibility traces persist? | answered (~1 s) |
| C5.001 | Can credit assignment work over 100+ ticks without BPTT? | open |
| C6.001 | When should structural growth replace weight change? | open |
| C7.001 | Does tag-based consolidation reduce forgetting? | open |

## D · Memory

| id | question | status |
|---|---|---|
| D1.001 | What is the capacity of a gated attractor working memory in items? | open |
| D2.001 | How does dentate separation scale with input similarity? | open |
| D3.001 | What replay ordering maximizes consolidation? | open |
| D4.001 | Does reverse replay help or hurt? | open |
| D5.001 | How to avoid catastrophic interference in a single region? | open |

## E · Control

| id | question | status |
|---|---|---|
| E1.001 | Is WTA action selection sufficient, or is a softmax needed? | open |
| E2.001 | How should exploration be scheduled? | open |
| E3.001 | Can a forward model be learned with the same local rule as prediction? | open |

## F · Perception

| id | question | status |
|---|---|---|
| F1.001 | Does prediction error fall monotonically as a hierarchy learns? | open |
| F2.001 | How to represent objects (bound features) in SDRs? | open |
| F3.001 | What is the minimal grid module set for 2D path integration? | open |

## G · Language

| id | question | status |
|---|---|---|
| G1.001 | Can sequence memory handle long-range dependencies without attention? | open |
| G2.001 | Is a hierarchical grammar learnable with prediction-only objectives? | open |
| G3.001 | How to get compositionality from sparse conjunctions? | open |

## H · Systems

| id | question | status |
|---|---|---|
| H1.001 | Sparse synapse layout: how much faster than dense at 2 % activity? | answered |
| H2.001 | Can columns be processed in parallel without locks? | answered (yes, independent) |
| H3.001 | How to checkpoint a growing synaptic structure? | open |

## I · Evaluation and safety

| id | question | status |
|---|---|---|
| I1.001 | Which benchmark suite is fair for non-transformer systems? | open |
| I2.001 | What ablations prove prediction (not memorization) drives accuracy? | open |
| I3.001 | How do we read the SDRs to explain a decision? | open |
| I4.001 | What are the failure modes under reward hacking? | open |

## J · Emotion, focus, body, consciousness

| id | question | status |
|---|---|---|
| J1.001 | Does scalar `ach` gain reproduce attentional selection in our regions? | implemented |
| J2.001 | Can valence be stored as extra SDR bits without breaking overlap? | open |
| J3.001 | Does interoceptive input improve value learning under delayed reward? | open |
| J4.001 | What arbitrates model-free vs model-based control? | open |
| J5.001 | Can global broadcast (GW-like) be an SDR OR over regions? | open |
| J6.001 | How to measure integration (IIT-like) in a sparse network? | open |
| J7.001 | Does high arousal help or hurt prediction? (expect inverted-U) | implemented |
| J8.001 | Can we induce mind-wandering by lowering `ach` and raising DMN gain? | open |
| J9.001 | Is a "heart rate" input better than an explicit valence input? | open |

## Process

1. Add questions as you hit a decision; never resolve silently.
2. Cite ids in code (`// Q:C3.001`) and in benchmark results.
3. Review 100 questions per session; prune `rejected` with a reason.
4. Prefer running an experiment over reading a paper.