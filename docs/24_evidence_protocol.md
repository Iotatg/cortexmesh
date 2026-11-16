# 24 · Evidence protocol: what counts as proof for a module

## The rule
A module is only "done" when it has numbers for all five questions, each with a control:

| question | measurement | control / baseline |
|---|---|---|
| Does it learn? | performance vs amount of data (learning curve) | unigram / frequency / random |
| Does old knowledge survive? | retention after new data | no-learning reference |
| Does it predict unseen data? | held-out or novel items | chance / shuffled labels |
| Is it small? | peak RSS and bytes per stored item | dense baseline of the same capacity |
| Does it transfer? | train on dataset A, test on B | same-dataset upper bound |

A number without its control is not evidence. A module that beats chance but loses to a
trivial baseline is not done.

## Results (all measured, `./build/bin/bench` and the tools below)

### SP — perception / feature learning
```
sp,stability=0.994,noise1=0.978,noise8=0.882,noise16=0.795
sp,classinfo_input=1.000,classinfo_sp=1.000
sp,pairwise_code_overlap=0.040,dead_columns=209,syn_per_col=32.0
sp,ram_matrix_kb=256
```
- Learns stable codes (0.994 repeat), degrades gracefully under noise (0.795 at 16 flips).
- Preserves class information: readout accuracy 1.000 both with and without SP, so the SP
  is not destroying the signal (control = input-only readout).
- Codes are nearly orthogonal across classes (0.040 pairwise overlap).
- **Fails:** 209/512 columns dead at 20 classes × 20 samples. Homeostasis kills columns
  when the input set is small; this is the capacity law again.
- RAM: 256 KB for 512 columns × 64 synapses.

### TM — sequence learning and prediction
```
tm,predict_acc=1.000,prev_baseline=0.016,freq_baseline=0.021
tm,burst_clean=0.8844,burst_corrupt=0.8815,segments=60160
tm,ram_segments_kb=2350
```
- Order-2 Markov source: next-symbol accuracy **1.000** vs 0.016 (previous symbol) and
  0.021 (most frequent). It learned the transitions, not the frequencies.
- **Fails:** burst rate does not separate clean from corrupted transitions (0.884 vs
  0.882). With 60k segments and a dense `prev` context, random contexts still find a
  matching segment. The burst anomaly signal needs sparse contexts and a segment cap
  that is not saturated.
- RAM: 2.3 MB of segments.

### Associative memory — learning, retention, composition
```
assoc,capacity,patterns=10..100,recall=1.000
assoc,capacity,patterns=200,recall=0.975
assoc,capacity,patterns=400,recall=0.868
assoc,retention,before=0.999,after_B=0.982
assoc,composition,overlap_X=1.000,overlap_Y=0.000
assoc,ram_matrix_kb=1024
```
- Learns 100 random pairs perfectly, 200 at 0.975, 400 at 0.868 — above the classical
  `0.15 N` correlation-matrix estimate, because inputs are sparse and random.
- Retention after a second list: 0.982 (no replay, orthogonal codes).
- **Fails composition:** probing `A|B` returns X (1.000) and not Y (0.000). OR-binding
  does not superpose two associations; this needs VSA-style bundling with cleanup, not
  top-k recall.
- RAM: 1 MB for a 512×512 float matrix.

### Language — streaming EMA model (D24)
```
xi=0.001: first-half 0.163, second-half 0.156, top1 0.160, top10 0.289, MRR 0.211, 15 MB
```
- Learns (0.16 top-1 vs 0.06 unigram), retains (0.163 → 0.156 across halves), predicts
  unseen text (held-out 15 %), and uses 15 MB for 1.74M tokens.
- Trained on wikitext-2 (train split) and evaluated on its own held-out tail; transfer to
  a different corpus is still unmeasured.

### Transfer across datasets (same model, no retraining)
Trained on the first 283k tokens of wikitext-2, then evaluated without learning:

| eval corpus | top1 | top3 | MRR | tokens |
|---|---|---|---|---|
| wikitext-2 tail (same domain) | 0.259 | 0.377 | 0.318 | 194k |
| TinyStories valid (children's stories) | 0.226 | 0.328 | 0.277 | 385k |

Cross-domain retains 87 % of same-domain top-1. The model transfers, but the gap is the
cost of a fixed 3k vocabulary (TinyStories has many OOV words). Sub-word units are the
fix, not a bigger matrix.

### Earlier modules (from their tools)
- Speech: forward 0.971, inverse 1.000, noise-8 0.912, permuted-label control 34/34.
- Curriculum: multi-hop 1.000, shuffled control 0.000, replay tradeoff measured.
- Audit: word↔concept overlap 1.50/24 (no leak), shuffled mapping 24/24.

## What is still missing

| capability | status | next measurement |
|---|---|---|
| memory | measured for SP/TM/assoc | capacity at 10× scale, interference curves |
| learning | measured (SP, TM, assoc, LM) | learning curve on a second dataset |
| reasoning | multi-hop only | 3-hop chains, contradiction detection |
| attention | not implemented | SDM hard-location memory as attention replacement |
| perception | SP noise robustness only | invariant recognition under transformations |
| language | wikitext + TinyStories transfer | sub-word units, third corpus |

## Rules for future modules
1. Every new module ships with a `bench` section and a control.
2. Report peak RSS, not just accuracy.
3. Report the failure, not only the win (dead columns, burst saturation, no composition).
4. Re-run `./build/bin/bench` and `ctest` before claiming a result.
5. A module that loses to a trivial baseline is not accepted, even if it is "brain-like".
## Ranking must exclude the catch-all class

`streamlm` first reported 0.226 top-1 on TinyStories. 22.5% of targets were out of vocabulary
and UNK was the most frequent output, so the model was mostly **predicting UNK**. Ranking over
known words only drops the score to 0.091 - still positive transfer, but a different claim.

Rule: whenever a catch-all class (UNK, other, none) exists, report accuracy **both** including
and excluding it, plus the catch-all rate. A score that is close to the catch-all rate is not
a result.
