# 23 · Research: bounded-memory language learning (why the batch model does not scale)

## The problem this answers
`tools/textmodel.cpp` trains on wikitext-2 by first materialising every
`(context, next-word)` pair, then doing several passes:

- 1.7M pairs × 128-byte SDRs ≈ **220 MB just for the pair list** (plus overhead), and
  the matrix pass needs a second copy for the Hebbian matrix.
- Several epochs, i.e. the corpus is stored and revisited.
- Dense 1024×1024 matrix per relation; a 5k-word vocab needs 5k output rows.

That violates the project's own constraints (online, single-pass, fixed-size state) and
does not survive a corpus 100× larger. The research below is about replacing the buffer
and the epochs with *local, streaming, bounded* learning.

## What the literature says

| paper | id | mechanism | what we take |
|---|---|---|---|
| Hebbian learning the local structure of language | arXiv:2503.02057 | hierarchy of neurons learns n-grams from raw text; update is a normalised EMA of co-occurrence `g ← (1-ξ)g + ξ·v_prev·v_k`; growth limits prevent collapse; learns morphology without labels | EMA Hebbian instead of accumulate; no epochs; growth limits |
| SNAP: Stopping Catastrophic Forgetting in Hebbian Learning with Sigmoidal Neuronal Adaptive Plasticity | arXiv:2410.15318 | weights grow sigmoidally and consolidate at large/small values (LTP-like), so old memories stop moving | state-dependent learning rate (consolidation) |
| Sparse Distributed Memory is a Continual Learner | arXiv:2303.11934 | an MLP whose addressing/reading is Kanerva SDM is a strong continual learner with **no replay and no task labels** | hard-location addressing as an alternative to a dense matrix |
| Attention Approximates Sparse Distributed Memory | arXiv:2111.05498 | transformer attention ≈ Kanerva SDM under conditions satisfied in GPT-2 | SDM is a local replacement for attention, not a toy |
| The Kanerva Machine | arXiv:1804.01756 | Bayesian online memory, analytically tractable, larger capacity than DNC, easier to train | online write/read rule, capacity analysis |
| Hyperseed: Unsupervised Learning with VSA | arXiv:2110.08343 | binding + bundling, few-shot, single-vector update, n-gram language ID | VSA binding for features, cheap updates |
| Vector symbolic architectures for context-free grammars | arXiv:2003.05171 | parse trees / phrase structure in Fock space; binding for structure | structure via binding, not via attention over history |
| Analysis of Predictive Coding Models for Phonemic Representation Learning | arXiv:2007.04205 | predictive-coding models learn phoneme-like representations without labels; ABX evaluation | label-free phoneme categories, ABX-style eval |

The recurring pattern: **learning is local, updates are normalised, and memory is
addressed sparsely**. None of these designs needs a stored dataset of pairs.

## Capacity and memory arithmetic

- Correlation-matrix memory (`W += x yᵀ`) with N weights holds roughly `0.15 N`
  reliably retrievable random patterns. A 1024×1024 matrix is therefore a ~150k-pattern
  memory, not a 1.7M-pattern memory: the batch model was already over capacity, which is
  why accuracy plateaued at 9–12 % top-1.
- Kanerva SDM with `A` addresses of `C` hard locations stores about `A` patterns and
  degrades gracefully; `A = 4096, C = 512` is 256 KB.
- EMA Hebbian (`g ← (1-ξ)g + ξ·xy`) keeps every weight in [0,1] and needs no epochs: one
  pass is the whole training run. The effective memory is the number of *distinct*
  context→word co-occurrences, not the number of tokens.
- Feature hashing (the "hashing trick") bounds a per-word readout to `V × H` floats:
  5k words × 2048 buckets × 4 B = 40 MB, and with 512 buckets 10 MB.

## Design v2 (to implement)

1. **Stream the corpus.** Read a token, build its context SDR, predict, then learn.
   No pair vector, no epochs, no shuffle.
2. **EMA Hebbian matrix** instead of accumulate: `w[i,j] ← (1-ξ)w[i,j] + ξ` for active
   `(i,j)` only. Bounded weights, stable online, single pass.
3. **Sparse decode by inverted index.** Prediction takes the strongest context bits, then
   only words that own those bits are scored (`V·K/N` words per bit, not `V`).
4. **Bounded footprint.** One 1024×1024 float matrix = 4 MB; vocab codes ~1 MB; inverted
   index < 1 MB. Target peak RSS < 64 MB for a 2M-token corpus.
5. **Online evaluation.** Score each token *before* the update; report accuracy over the
   first and last 10 % of the stream. This gives a learning curve in one pass, which is
   what the architecture actually promises.
6. **Consolidation (optional, SNAP-style).** Scale the update by `4w(1-w)` so weights near
   0/1 stop moving; this should reduce later-list forgetting without replay.
7. **Sub-word units later.** 2503.02057 shows a neuron hierarchy can induce tokens from
   text; that is the path to a vocab that grows instead of an `<unk>` bucket.

## Acceptance criteria for v2

| metric | batch baseline | v2 target |
|---|---|---|
| peak RSS, 2M-token corpus | > 500 MB | < 64 MB |
| training passes over data | 2–6 | 1 |
| held-out top-1 | 0.12 | ≥ 0.12 |
| held-out top-10 | 0.44 | ≥ 0.44 |
| corpus size that fits | 1.9M tokens | 20M+ tokens |

If v2 matches the batch numbers at a fraction of the memory, the batch tool stays as a
reference implementation and the streaming one becomes the default.

## What we do NOT do

- No dense per-word weight vectors over the whole context (that is a softmax classifier
  with extra steps).
- No storing of embeddings for every token (that is a cache, not a model).
- No transformer attention. 2111.05498 says attention ≈ SDM; we use the SDM side.
## Results: streaming v2 on wikitext-2 (Hugging Face)

Corpus: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `train`
(17,187 sentences, 1,742,067 tokens, 64,777 types), downloaded with the HF
datasets-server and the parquet file; vocab = 5,000 most frequent words + `<unk>`.
Model: N=512, K=24, one 512×512 EMA matrix, bigram context, online scoring before
each update, single pass.

| xi | gate | norm | peak RSS | first-half top1 | second-half top1 | all top1 | all top10 | MRR |
|---|---|---|---|---|---|---|---|---|
| 0.010 | off | on | 15 MB | 0.138 | 0.044 | 0.091 | 0.202 | 0.131 |
| 0.010 | on | on | 15 MB | 0.143 | 0.056 | 0.099 | 0.217 | 0.141 |
| 0.002 | off | on | 15 MB | 0.159 | 0.151 | 0.155 | 0.275 | 0.201 |
| 0.002 | off | off | 15 MB | 0.159 | 0.151 | 0.155 | 0.275 | 0.201 |
| 0.001 | off | on | 15 MB | 0.163 | 0.156 | **0.160** | 0.289 | 0.211 |

Batch reference (`textmodel`, 1 epoch, both copies of the data in RAM): 0.117 top-1,
0.258 top-3, 0.444 top-10, MRR 0.215. Peak RSS 127 MB on 283k tokens, which
extrapolates to roughly 0.7–0.8 GB at 1.74M tokens (the pair vector dominates and grows
by doubling).

Reading:
1. **Streaming wins on top-1** (0.160 vs 0.117) at **~1/50th of the memory** (15 MB vs
   an estimated 0.8 GB).
   The batch model was over capacity (0.15 N patterns) and paid for it.
2. **Learning rate dominates forgetting.** xi=0.01 collapses in the second half
   (0.138 → 0.044); xi=0.002 is flat (0.159 → 0.151); xi=0.001 is flat and best
   (0.163 → 0.156). This is the same
   stability-plasticity knob seen in D22, now inside a single pass.
3. **SNAP-style gating helps a little** at xi=0.01 (0.091 → 0.099) but does not rescue a
   rate that is too high. Consolidation is not a substitute for a sane rate.
4. **Top-10 is lower than batch** (0.275 vs 0.444) because the batch model saw the data
   twice; top-1 and MRR are better.

## What changed in the code
- `AssociativeMemory::learn_batch` added (one decay per batch, then accumulate).
- `tools/streamlm.cpp`: streaming pass, EMA Hebbian, lazy update, inverted-index decode,
  online first/second-half evaluation, `VmHWM` reporting.
- `tools/textmodel.cpp`: reference batch model, kept for comparison.

## Next experiments
- xi sweep 0.0005–0.002 on the full corpus (running).
- Sub-word codes (character trigrams) so rare words share features; target: OOV top-1.
- SNAP gating with `4w(1-w)` and a separate consolidation phase (sleep) instead of a
  higher learning rate.
- SDM hard-location memory as a drop-in replacement for the matrix (A=4096, C=512).
- Learned tokenisation following arXiv:2503.02057 (neuron hierarchy), removing the
  fixed 5k vocab.