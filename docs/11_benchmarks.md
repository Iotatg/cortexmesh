# 11 · Benchmarks (measured)

Environment: x86-64 Linux, GCC 12, `-O2`/`Release`, single thread. Reproduce with
`ctest` and `./bin/brain_cli seq`. Every number below was produced by the code in this repo.

## B1 · SDR correctness

| Property | Target | Measured |
|---|---|---|
| `count()` after construction | `k` | exact |
| `overlap(a,a)` | `k` | exact |
| Two random codes (n=1024, k=40) | ~1.6 by chance | 0–3 |
| `jaccard(a,a)` | 1.0 | 1.0 |
| `a | b` size | `|a|+|b|-|a∩b|` | exact |

## B2 · Spatial pooler

Config: `input=1024, columns=4096, synapses=128, active=20`.

| Metric | Target | Measured |
|---|---|---|
| Code stability (overlap of repeated presentations, 30 epochs) | > 0.95 | **0.995** |
| Discrimination (different inputs) | < 0.2 | **0.0** |
| Dead columns after 300 inputs | 0 | **0** |
| Throughput | — | **1.8 ms / activation** |
| Output sparsity | 2 % of columns | 20 / 4096 = 0.49 % (configurable) |

Note: boosting is what keeps columns alive; with `boost_max=1.0` some columns go silent
(the classic SP failure mode). This is a tuning result, not a bug.

## B3 · Temporal memory (pure)

Config: `columns=64, cells=16, threshold=1`, sequence length 32, 30 epochs.

| Metric | Target | Measured |
|---|---|---|
| Next-step prediction accuracy | > 0.9 | **1.000** |
| Bursts on learned transitions | 0 | **0** |
| Anomaly on unseen column | 1 | **1** (burst, `predicted_active=0`) |

## B4 · SP + TM (sequence through an encoder)

Config: `input=1024, columns=4096, cells=16, active=20`, 32 items, 30 epochs.

| Metric | Target | Measured |
|---|---|---|
| Code drift over training | < 0.05 | **0.005** (overlap 0.995) |
| Prediction accuracy | > 0.9 | **1.000** |
| Bursts on learned transitions | 0 | **0** |

This is the first real result: **the region learns the sequence, predicts it, and flags
novel transitions**, with local rules only.

## B5 · Region (composed unit)

Config: `input=1024, columns=1024, cells=16, active=20`, 8 items, 15 epochs.

| Metric | Target | Measured |
|---|---|---|
| Prediction accuracy | > 0.8 | **1.000** |
| Anomaly on unseen input | > 0.5 | **1.0** |
| Segments grown | > 0 | > 0 |

## B6 · Known failure modes (measured, to fix)

| Failure | Symptom | Root cause | Status |
|---|---|---|---|
| Segment explosion | training slows to minutes | unbounded growth on every burst | fixed (`max_segments_per_cell=16`) |
| Dead columns | code collapses to a few columns | `boost_max` too low | fixed (`boost_max=1.0` default, boosting disabled) |
| Code drift | prediction collapses to ~0.15 | column count too small for item count | fixed by capacity rule: `columns ≥ 40 × items` |
| Correlated codes | different inputs map to same columns | weak RNG (xorshift low bits) | fixed (splitmix64, high bits) |
| Prune kills columns | accuracy collapses after `prune` | homeostat targeted current size, rescaled survivors down | fixed (nominal target, connectivity floor) |
| Overtraining drift | accuracy falls after convergence | permanent weights saturate, columns drift | train to convergence only (D10) |
| Interleaved replay | replay makes A worse than no replay | A and B compete for the same columns | replay in a separate phase (X1) |
| Intermittent bursts | ~20 % bursts on trained transitions | winner-only active state after prediction; segments grown from 16-cell bursts | needs predictive cells in prev state (D12) |
| Apical gain hurts | feedback lowers accuracy | column-level bias pulls wrong columns | feedback must be uncertainty-gated (D11) |
| SP invariant collapse | position changes invisible, jump2 burst 0 | constant sub-codes dominate top-k | single token per state (D13) |
| Representation drift | prediction collapses | winner boosting reshuffles codes | fixed (boosting off, D2) |
| Value interference | untrained action scores ~half | conjunction readout shares state bits | fixed (per-action heads, D6) |
| Single-cell segments | threshold 2 never met | `new_segment_min` > active cells in `prev` | fixed (test config) |

## B7 · Capacity rule (measured, phase 1.5)

For `I` distinct items to be separated without interference, at 20/1024 active columns,
64 synapses and 16 cells per column:

```
columns ≥ 32 × I
```

Measured: 8 items need 128 columns (8× is 0.375, 16× is 1.000); 32 items need 1024–2048
(16× is 0.719, 32× is 1.000); 64 items need 4096 (32× is 0.781, 64× is 1.000).

At larger item counts 32× stops being enough (X4):

| items | 32× | 64× |
|---|---|---|
| 64 | 0.953 | 0.969 |
| 128 | 0.930 | 0.969 |
| 256 | 0.812 | 0.988 |

So: `columns ≥ 32 × items` for `items ≤ 64`, `≥ 64 × items` above that.
Full table: `13_discoveries.md` D1 and X4.

Below this, columns are shared across items, transitions conflict, and accuracy collapses
(measured 0.14 at `columns=256, I=8, active=12`).

## B8 · Next benchmarks to add

1. Variable-order sequences (same column after different predecessors).
2. Anomaly ROC on corrupted inputs (bit flips) vs. clean.
3. Catastrophic forgetting across more than 2 lists (A, B, C).
4. Replay with 64× columns and 64 items (does end10 hold?).
5. Hierarchical denoising: can L2's prediction select the clean L1 code from noisy input?
6. Prune + relearn as a standard consolidation pass, measured across column counts.
7. Anomaly AUC: clean vs 1–8 bit flips, per corruption level (burst rate exists, ROC not yet).
8. Fix intermittent bursts by including predictive cells in the distal context.
9. Two-step segment growth for variable-order context (D12: 0.6, want 1.0).
10. Event-window detector for rare glitches (pass_wall, food_dup), not rate.
11. Action-conditioned forward model `(state, action) -> next state`.
12. Richer appearance stream (multi-scale patch, food proximity) for object events.
13. Articulatory + audio channel for language (sensorimotor loop).
14. Theta-gamma phase coding for order instead of shift binding.
15. Learn concepts from cross-situational exposure instead of hand-coded tables.
16. Word-level production: phone sequence -> articulation -> audio.
17. Co-articulation and learned auditory categories.
18. Nested theta-gamma (slow phase for position, fast for phone).
19. Learned syllabification from transition statistics.
20. Auditory categories from prediction error.21. Knowledge induction: multi-hop chains, feature-bundle interference, replay stability.
22. Ordered speech: nested theta-gamma vs flat phases, co-articulation, held-out words.
23. DONE in D22: interleaved replay (1:1) vs block replay on retention.
24. Pattern separation before the fact matrix (orthogonalising similar entities).
26. Replay ratio sweep (1:1, 1:4, 4:1) and replay rate as a function of set size.
27. DONE in D24: streaming EMA LM on wikitext-2 (1.74M tokens, 15 MB).
28. Sub-word codes so rare words share features (OOV top-1).
29. SDM hard-location memory replacing the dense matrix.
30. Learned tokenisation (arXiv:2503.02057) instead of a fixed 5k vocab.
25. Fact revision and contradiction detection (non-monotonic updates).

## Benchmark 31 · TM with a learnable source (order-1, 64 contexts)

| metric | value |
|---|---|
| prediction accuracy | 0.878 |
| prev-symbol baseline | 0.009 |
| frequency baseline | 0.051 |
| burst fraction, clean stream | 0.412 |
| burst fraction, corrupt stream | 0.521 |
| segments | 30,080 |
| segment RAM | 1.2 MB |

## Benchmark 32 · Streaming LM, known-word ranking

| set | oov | inv top-1 | inv top-10 | mrr |
|---|---|---|---|---|
| wikitext tail | 0.260 | 0.105 | 0.183 | 0.105 |
| TinyStories | 0.225 | 0.091 | 0.151 | 0.093 |

Peak RSS 15 MB, 15k tokens/s single pass.

## Benchmark 33 · Character-level LM (no OOV)

| set | top-1 | top-5 | MRR | oov |
|---|---|---|---|---|
| train all (1.73M chars) | 0.338 | 0.653 | 0.478 | 0.000 |
| TinyStories (transfer) | 0.438 | 0.622 | 0.533 | 0.000 |

Baselines: prev char 0.020, unigram 0.197. Context k=2 (k=3 0.250, k=4 0.231). RSS 11 MB.

## Benchmark 34 · SDM vs dense associative memory

| metric | SDM (A=4096, C=1024) | dense (1024x1024 float) |
|---|---|---|
| recall >= 0.9 up to | 1500 patterns | 3200 patterns |
| retention | 0.998 | 0.982 |
| noisy cue 50% | 0.509 | 0.970 |
| RAM | 1.03 MB | 4.19 MB |
| correlated-code recall | 1.000 | 0.799 |
| control (shuffled) | 0.048 (chance 0.039) | - |

## Benchmark 35 · TM novelty AUC

| signal | random | skip | stutter | pooled |
|---|---|---|---|---|
| burst fraction | 0.753 | 0.687 | 0.567 | 0.666 |
| surprise EWMA | 0.817 | 0.745 | 0.572 | 0.711 |
| shuffled control | 0.510 | 0.510 | 0.510 | 0.510 |

At 1% FPR: TPR 0.047. Windowed (200 steps): clean 0.41-0.43 vs corrupt 0.61.

## Benchmark 36 · Revision and hop depth

| m (contradictions) | recall old | recall new |
|---|---|---|
| 0 | 1.000 | 0.000 |
| 5 | 0.125 | 0.875 |
| 10 | 0.000 | 1.000 |
| 20 | 0.000 | 1.000 |

| depth | isolated matrix | shared matrix | shuffled control |
|---|---|---|---|
| 1 | 1.000 | 0.667 | 0.031 |
| 2 | 1.000 | 0.458 | 0.049 |
| 3 | 1.000 | 0.625 | 0.038 |
| 4 | 1.000 | 0.500 | 0.045 |

## Benchmark 37 · Context-free retrieval (no window)

| test | value |
|---|---|
| capacity 1k / 5k / 20k items | 0.970 / 0.515 / 0.183 |
| age 0 / 256 / 1024 recall | 0.912 / 0.912 / 0.894 |
| window 64 at age 256 | 0.000 |
| chain depth 1-5 | 1.00 / 0.75 / 0.09 / 0.02 / 0.00 |
| old item in new context | 0.692 (wrong 0.054) |
| state RAM | 1 MB, independent of age |

## Benchmark 38 · Cleanup memory (denoise)

| metric | value |
|---|---|
| exact denoise, 0-12 flips of 16 | 1.000 |
| exact denoise, 14 flips | 0.000 |
| 5-hop chain raw / cleaned | 0.000 / 0.990 |
| codes stored / RAM | 2600 / 512 KB |
| latency per clean, 4096 items | 29.7 us |

## Benchmark 39 · Read-out power sweep (capacity)

| load (pairs) | p=1 | p=2 | p=3 |
|---|---|---|---|
| 500 | 0.999 | 0.922 | 0.685 |
| 5000 | 0.512 | 0.407 | 0.308 |
| 20000 | 0.188 | 0.175 | 0.159 |

| case | p=1 | p=2 | p=3 |
|---|---|---|---|
| shared 8-bit role tag | 0.213 | 0.053 | 0.053 |
| A then B, recall A | 1.000 | 0.998 | 0.865 |
| shuffled-value control | 0.025 | 0.025 | 0.028 |

Verdict: a super-linear read-out is worse at every load and on every control. Negative result.

## Benchmark 40 · Cleanup memory scaling

| codes stored | exact denoise | us per clean | RAM |
|---|---|---|---|
| 1,000 | 1.000 | 14.2 | 0.1 MB |
| 10,000 | 1.000 | 136.8 | 1.2 MB |
| 100,000 | 1.000 | 1469 | 12.2 MB |
| 500,000 | 1.000 | 9229 | 61.0 MB |

Accuracy is flat at 1.000 across 500x growth; latency and RAM grow linearly (128 bytes per
code, i.e. the sparse code itself). RAM is bounded and predictable; the latency is the
binding constraint for online use, which motivates a candidate-generation stage.

## Benchmark 41 · Cleanup decode inside the language model

| decoder | top-1 | top-3 | top-10 | MRR |
|---|---|---|---|---|
| linear bit-score | 0.0315 | 0.0481 | 0.0666 | 0.0440 |
| + cleanup, lambda = 0.5 | 0.0272 | 0.0412 | 0.0547 | 0.0374 |
| + cleanup, lambda = 1.0 | 0.0126 | 0.0220 | 0.0425 | 0.0383 |

Verdict: cleanup re-ranking is worse at every lambda. Negative result (see D37).

## Benchmark 42 · Character model on real code

| model | next-char top-1 | top-5 | MRR | exact 16-char | mean prefix |
|---|---|---|---|---|---|
| N=256 ctx=2 | 0.3010 | 0.4824 | 0.3969 | — | — |
| N=256 ctx=4 | 0.3525 | 0.5526 | 0.4485 | — | — |
| N=256 ctx=6 | 0.3728 | 0.5847 | 0.4727 | — | — |
| N=256 ctx=8 | 0.3758 | 0.5966 | 0.4802 | — | — |
| N=256 ctx=12 | 0.3821 | 0.6072 | 0.4896 | — | — |
| N=512 ctx=8 | 0.3919 | 0.6128 | 0.4981 | 0.0010 | 0.725 |
| N=512 ctx=12 | 0.3846 | 0.6281 | 0.5008 | 0.0008 | 0.881 |
| unigram baseline | 0.3194 | — | — | — | — |

Corpus: 573 Python stdlib files (6.6 MB) -> 56 held-out files (585,471 chars), alphabet 95.

## Benchmark 43 · Weight saturation (the full-corpus collapse)

| xi | tokens | mean w | frac > 0.9 | inv_top1 |
|---|---|---|---|---|
| 0.005 | 8M | 0.9902 | 0.9694 | 0.0368 |
| 0.0005 | 8M | 0.7797 | 0.5228 | 0.0808 |
| 0.005 | 86.5M | — | — | 0.0039 |

Verdict: the matrix saturates; the 86.5M-token run collapses to chance in its second half.

## Bench 44 · bounded updates on the language matrix (3M tokens, wikitext-103, V=10000)

Same corpus, same xi, same architecture; only the update rule changes.

| rule | inv_top1 | top3 | mean w | frac > 0.9 | frac at cap |
|---|---|---|---|---|---|
| 0 plain Hebb | 0.0714 | 0.0702 | 0.9433 | 0.8237 | - |
| 1 importance-scaled (lam=10) | **0.0849** | 0.0772 | 0.7401 | 0.3609 | - |
| 2 covariance (mean-subtracted) | 0.0718 | 0.0705 | 0.8914 | 0.7654 | - |
| 3 importance + hard cap 0.6 | 0.0272 | 0.0272 | 0.5643 | 0.0000 | 0.7526 |
| 4 importance + soft target 0.7 | 0.0732 | 0.0701 | 0.6002 | 0.0000 | 0.0000 |
| 5 importance + column renorm 0.7 | 0.0765 | 0.0726 | 0.7167 | 0.3527 | 0.5894 |
| 6 importance + SHY downscale (0.82 on weakest 80%) | 0.0765 | 0.0719 | 0.2363 | 0.2000 | 0.2000 |

At 8M tokens: rule 1 = 0.0806 (frac > 0.9 = 0.6084), rule 6 = 0.0744 (frac > 0.9 = 0.2000).

Verdict: the importance term is the only change that helps (+19 %). All three ceilings remove
saturation and all three lose accuracy, because once many synapses share a value the frequency
normaliser decides the ranking instead of the content. Ratio-preserving renormalisation (rule 5)
preserves within-column ratios but still ties an already-flat column. SHY (rule 6) removes the most
saturation (36% -> 20%) and still loses 10% accuracy.

Verdict: at 3M-8M tokens saturation is **not** the binding constraint. Every mechanism that reduces it
also reduces accuracy, because the synapses that saturate are the frequent pairs that carry the signal.
The binding constraint is the representation, not the weight matrix. No update rule will fix it.

## Bench 45 · hashed n-gram cache vs Hebbian matrix (wikitext-103, 86.5M tokens)

| system | bytes | top1 | top3 | top10 | mrr | tokens/s |
|---|---|---|---|---|---|---|
| streamlm matrix, best rule (1) | 3 MB | 0.0039 | - | - | - | 5,500 |
| streamlm matrix, 8M tokens | 3 MB | 0.0663 | 0.0708 | 0.0830 | 0.0726 | 5,500 |
| cachelm order 3, 8M tokens | 48 MB | 0.1079 | 0.1600 | 0.1778 | 0.1344 | 135,000 |
| cachelm order 3, 86.5M tokens | 48 MB | **0.1100** | **0.1636** | **0.1816** | **0.1372** | 474,000 |

Verdict: the cache holds its accuracy from 8M to 86.5M tokens (0.1079 -> 0.1100) at a fixed 48 MB,
while the matrix collapses to 0.0039. Accuracy now costs memory (48 MB of slots vs 3 MB of weights),
not accumulation. This is the design CortexMesh should build on.

## Bench 46 · cache size vs accuracy on the full 86.5M-token corpus (order 3, 3 continuations)

| bytes | top1 | top3 | top10 | mrr |
|---|---|---|---|---|
| 6 MB | 0.0761 | 0.1108 | 0.1190 | 0.0930 |
| 12 MB | 0.0865 | 0.1272 | 0.1379 | 0.1066 |
| 24 MB | 0.0978 | 0.1449 | 0.1589 | 0.1214 |
| 48 MB | 0.1100 | 0.1636 | 0.1816 | 0.1372 |

Roughly +0.011 top-1 per doubling. The 3 MB Hebbian matrix scores 0.0039 on the same corpus, so even
the 6 MB cache is 19x better. Accuracy is bought with memory, not with accumulation.

## Bench 47 · matrix + cache interpolation (500k tokens)

| system | top1 | top3 | top10 | mrr |
|---|---|---|---|---|
| matrix only | 0.0718 | 0.0733 | 0.0854 | 0.0769 |
| matrix + cache (beta 0.5) | 0.0897 | 0.1127 | 0.1280 | 0.1054 |

Verdict: blending wins on every metric. The matrix generalises, the cache remembers, and the
interpolation gets both. Throughput drops from 5.5k to 2.5-3.3k tokens/s, which is the next thing to
optimise (merge the cache into `wscore` instead of scanning the candidate list).

## Bench 48 · minimal-pair grammar (forced choice, 50% = chance)

`tools/grammar.cpp`, 28 hand-written pairs, 6 phenomena. Matrix only, cache blended at beta 0.5.

| tokens | rule | OVERALL |
|---|---|---|
| 2M | plain Hebb | 60.7% |
| 20M | plain Hebb | **3.6%** |
| 20M | importance + SHY | 57.1% |
| 86.5M | importance + SHY | 57.1% |
| 20M | importance + SHY, skip context 1,2,3,4,6,8 | 57.1% |

Verdict: grammar is not learned. More data collapses the model (3.6%, below chance) until the update
rule is bounded, and even then it plateaus at 57%, which is a shallow-collocation bias rather than
syntax. The architecture has no mechanism for a dependency that spans more than the context window.

## Bench 49 · context mechanism vs minimal-pair grammar (20M tokens, matrix only)

| context | OVERALL | agreement_long | aux |
|---|---|---|---|
| fixed slots 1,2,3 | 57.1% | 66.7% | 25.0% |
| trace k=48, gamma=0.9 | 60.7% | 66.7% | 50.0% |
| trace k=48, gamma=0.95 | 60.7% | 66.7% | 50.0% |
| skip slots 1,2,3,4,6,8 | 57.1% | 50.0% | 50.0% |

Verdict: state beats window (+3.6 points), skip positions do not help, and every variant sits at
60-61% where chance is 50%. Grammar needs a representation with explicit features, not a better
context.

## Bench 50 · minimal-pair grammar after the head and feature-table fixes

Configuration: the whole 335k-token `agree_train.txt`, 6 slots (offsets 1-6), hybrid feature code
(`code_mode=4`, `feat_repeat=8`),
xi=1e-4, importance + SHY, head stack on (`head_mode=1`, `head_w=4`), `slot_head_w=0.25`.

| pair set | pairs | correct | acc |
|---|---|---|---|
| tuned (`MakePairs`) | 20 | 20 | **100.0%** |
| held-out (`agree_heldout.txt`) | 19 | 19 | **100.0%** |
| hard (`agree_hard.txt`) | 19 | 19 | **100.0%** |
| **all three** | **58** | **58** | **100.0%** |

Robustness of the held-out set:

| knob | values | held-out |
|---|---|---|
| xi | 1e-5, 1e-4, 1e-3 | 100% at all three |
| slot_head_w | 0.0, 0.1, 0.25, 0.5, 1.0 | 100% at all five |

The corpus is 335k tokens and every run consumes all of it, so `max_tokens` above 335k changes
nothing; the data-size sweep that looked flat was really one data size repeated.

Verdict: with a head that identifies the subject, a feature table that covers the lexicon, and a
readout that can see the feature bits, the matrix learns number agreement on clean data and
generalises to unseen nouns, verbs and prepositions. `slot_head_w` is not load-bearing, which is
evidence that the gain is the head, not the suppression.

## Bench 51 · the same mechanism on real text

| corpus | tokens | overall |
|---|---|---|
| synthetic | 335k | 100.0% (20/20) |
| wikitext-103 | 3M | 64.3% (18/28) |
| wikitext-103 | 20M | 57.1% (16/28) |

Verdict: real text does not benefit from more data and stays near the 57% plateau of Bench 48.
The clean corpus isolates agreement; wikitext does not. Extracting the rule from noisy text is the
open problem, not the head mechanism.
