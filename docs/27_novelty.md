# 27 · Novelty detection: can the temporal memory flag the unexpected?

## Question

D26 left an open item: the temporal memory predicts an order-1 source well (next-symbol
accuracy 0.878) but its burst signal separated clean from corrupt input only weakly
(0.412 vs 0.521). Is burst, or any other quantity derivable from the TM output, a usable
novelty detector?

This is measured as an ROC problem, not as two mean rates. A raw rate cannot answer the
question that matters: at an acceptable false-positive rate, how often is a genuinely
novel event caught?

## Setup

Copied from `BenchTm` in `tools/bench.cpp` so the numbers are comparable:

| item | value |
|---|---|
| symbols | 64, `Encoder(512, 20, seed 9)` |
| source | order-1, 3 successors per symbol, `Rng(23)` |
| TM columns x cells | 512 x 8 |
| max segments / cell | 8 |
| activation threshold | 6 |
| new segment min | 6 |
| training steps | 30000 |
| evaluation | learning gain 0 (frozen model) |

Command and cost:

```
cd /workspace/brain && cmake -S . -B build_nov -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build_nov -j4 --target novelty
./build_nov/bin/novelty
```

14.6 s wall (11.1 s user), peak RSS 9 MB, exit 0, no warnings under
`-Wall -Wextra -Wpedantic`.

## Signals

Every signal is computed in the tool from the returned `TMStep` (public API only). Let
`B` be the number of bursting columns out of 20 active input columns, `A` the set of
active input columns, `P` the columns covered by `predictive`, and `C` the context the
TM used (previous active | previous predictive).

| name | definition | intent |
|---|---|---|
| `burst_fraction` | `B / |A|` | baseline: how much of the input was unpredicted |
| `pred_error` | `1 - |P intersect A| / |A|` | fraction of the actual input that was not predicted |
| `unexpectedness` | `1 - predicted_active / active_cells` | cell-level version of the same |
| `surprise_ewma` | EWMA of `pred_error`, alpha 0.2 | persistence: one odd event does not fire alone |
| `surprise_slow` | EWMA of `pred_error`, alpha 0.05 | slower persistence control |
| `context_mismatch` | mean over `A` of `1 - min(1, best_matching_synapses / 6)`, using `segments()` and `C` | graded match strength, not a binary burst flag |
| `surprise_win20` | trailing mean of `pred_error` over 20 steps | block-rate view |

### The three "prediction" signals are the same signal

`step()` only evaluates segments for columns that are present in the input, and it writes
one predictive cell per non-bursting input column. Therefore `P` is always a subset of
`A`, and for every non-bursting column exactly one active cell is predictive, while a
bursting column contributes 8 active cells and no predictive cell. With `B` bursting
columns out of 20:

```
burst_fraction  = B / 20
pred_error      = 1 - (20 - B) / 20 = B / 20
unexpectedness  = 1 - (20 - B) / (20 + 7B) = 8B / (20 + 7B)
```

All three are strictly monotone in `B`, so they induce identical rankings and identical
AUC. The measured AUCs are equal to three decimals, which confirms the algebra. This is a
property of the library, not a bug in the tool.

The consequence is a real limitation: `predictive()` cannot express a prediction that
failed because a column was predicted and did not arrive. That would require reading
`segments()` for all 512 columns every step and testing them against `C`; the tool
implements that test only for the active columns (`context_mismatch`), which is why it
stays near chance.

## Anomaly injection

Three types, injected independently into the same trained model (frozen), each in its own
4000-step stream after a 200-step clean warm-up, with per-step probability 0.10:

| type | injected input |
|---|---|
| `random` | uniform random symbol (novel) |
| `skip` | the successor of the expected successor (omission of one symbol) |
| `stutter` | a repeat of the previous symbol |

Labels: clean transition 0, injected step 1. Pooled: 12000 steps, 10779 clean, 1221
anomalies (10.2 %).

## Results

### Clean statistics (3000-step held-out clean stream)

```
predict_col_recall=0.604  predict_any=0.931  predict_step_acc=0.575
burst=0.3961  anomaly_score_dev=0.00e+00  segments=28321
```

`anomaly_score_dev` is the maximum difference between the library's `anomaly_score()` and
the tool's `burst_fraction` over the stream; it is exactly zero, so the baseline signal is
the library's own anomaly measure.

The mean burst fraction reproduces D26's 0.412 (0.396 here, different stream position).
`predict_any` is 0.931, above the bench's 0.878. The bench measures accuracy on a stream
that starts after 2000 corrupted transitions, which leaves the region in a disturbed
state; the number here is measured on a clean stream after a 100-step warm-up. The gap is
consistent with that difference but is not a controlled comparison.

### ROC AUC, all signals x anomaly types

| signal | random | skip | stutter | pooled |
|---|---|---|---|---|
| `burst_fraction` | 0.753 | 0.687 | 0.567 | 0.666 |
| `pred_error` | 0.753 | 0.687 | 0.567 | 0.666 |
| `unexpectedness` | 0.753 | 0.687 | 0.567 | 0.666 |
| `surprise_ewma` (a=0.2) | 0.817 | 0.745 | 0.572 | 0.711 |
| `surprise_slow` (a=0.05) | 0.687 | 0.675 | 0.534 | 0.629 |
| `context_mismatch` | 0.524 | 0.539 | 0.512 | 0.525 |
| `surprise_win20` | 0.623 | 0.632 | 0.518 | 0.591 |

Mean AUC over the three types: `burst_fraction` 0.669, `pred_error` 0.669,
`unexpectedness` 0.669, `surprise_ewma` 0.711, `surprise_slow` 0.632,
`context_mismatch` 0.525, `surprise_win20` 0.591.

Best signal: `surprise_ewma`, mean AUC 0.711.

### Control: shuffled labels

The pooled labels are permuted and the AUC recomputed:

```
burst=0.507 pred_error=0.507 unexpectedness=0.507 surprise_ewma=0.507
surprise_slow=0.515 context_mismatch=0.508 surprise_win20=0.509
```

All at chance, as required. The AUCs above are not an artefact of the scoring code.

### Time-constant sensitivity

EWMA alpha swept offline over the saved `pred_error` sequence:

| alpha | 0.02 | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 | 0.80 |
|---|---|---|---|---|---|---|---|
| pooled AUC | 0.573 | 0.629 | 0.678 | 0.711 | 0.718 | 0.712 | 0.686 |

The optimum is broad and flat around 0.2-0.5. The main table keeps the a-priori
alpha = 0.2; 0.35 would add 0.007 AUC, which is inside run-to-run noise.

### Operating point at a fixed false-positive rate

`surprise_ewma`, pooled stream:

| target FPR | threshold | TPR | measured FPR |
|---|---|---|---|
| 1 % | 0.5495 | 0.047 | 0.0098 |
| 5 % | 0.5309 | 0.129 | 0.0496 |
| 10 % | 0.5181 | 0.234 | 0.0992 |

At the requested 1 % FPR the detector catches 4.7 % of injected anomalies. That is the
honest headline: the signal ranks anomalies above clean input (AUC 0.71) but the two
distributions overlap so heavily that a low-false-alarm operating point is useless.

### Sequential test: 2000 clean, then 200 corrupt

Mean `surprise_ewma` per 200-step bin (first 10 bins clean, last bin all-random):

```
clean: 0.415 0.425 0.420 0.426 0.412 0.415 0.406 0.432 0.423 0.416
corrupt: 0.609
```

The clean bins span 0.406-0.432 and the corrupt bin sits at 0.609, with no overlap. At
*block* granularity the separation is clean; at single-step granularity it is not.

## Why the per-step signal is weak

1. **The region is saturated.** 28321 segments against a hard cap of
   512 x 8 x 8 = 32768 (86 % full). D26 rule 1 says a sequence region needs
   `cells_per_column x max_segments_per_cell >= distinct contexts` with headroom; here the
   cap is nearly reached, so segments are evicted and re-grown, and a large fraction of
   clean transitions still burst.
2. **Clean input bursts 40 % of the time.** With a clean mean of 0.396 and a corrupt mean
   near 0.6, no threshold can separate them per step. The signal is not broken, it is
   sitting on top of a noisy baseline.
3. **Graded matching does not help.** `context_mismatch` reads the actual synapse counts
   and still scores 0.525, i.e. chance. When segments are dense and the context is 20-40
   active cells, a random column finds a matching segment as easily as the correct one.
4. **Slow smoothing hurts.** A 20-step window (0.591) and a 0.05 EWMA (0.632) are both
   worse than 0.2, because at 10 % anomaly density every window is already contaminated.
   Persistence only pays when anomalies arrive in runs, which is exactly the sequential
   test above.

## Verdict: partly

- **As a per-step detector: no.** Best AUC 0.711, and only 4.7 % TPR at 1 % FPR
  (12.9 % at 5 %, 23.4 % at 10 %). A perfect predictor would still burst on novel input,
  and this TM bursts on 40 % of clean input, so the anomaly and clean distributions are
  not separable at low false-alarm rates.
- **As a block / rate detector: yes, with a caveat.** Over 200-step windows the clean
  signal sits at 0.406-0.432 and fully corrupt input at 0.609, with no overlap. A
  window-rate threshold near 0.5 would flag sustained corruption. The caveat is that this
  is one 200-step corrupt block; the per-step AUC says the margin is not large.
- **As a ranking signal: partly.** AUC 0.82 for novel symbols and 0.75 for omissions is
  useful for triage (which events to inspect first), not for hard decisions.

The fix is not a better scoring function on the same output. It is capacity and sparsity:
more segments per cell (or fewer contexts), so the clean burst floor drops. With the
current 28k-segment saturated region the TM is a mediocre novelty detector, and the raw
burst fraction is no better than the prediction error it is algebraically equal to.

## Limitations

1. `predictive()` is defined only over the input columns, so a true prediction error
   (predicted column that never arrived) is not observable through the public API. The
   tool documents this instead of changing the library. A full 512-column generative check
   via `segments()` would add roughly 2 x 10^8 synapse tests per 1000 steps and does not fit
   in the sub-30 s budget; it is left as future work.
2. `context_mismatch` tests the same context the TM used, reconstructed from the previous
   `TMStep`, so it matches the library's internal `prev_active | prev_predictive`
   construction but is not guaranteed bit-identical if the library changes.
3. One seed, one source. The alpha sweep and the AUC values are single-run numbers.
4. The 200-step corrupt block is a single block; a per-block ROC over many blocks is the
   next measurement, together with a repeat at 64x columns per D26 rule 1.
## Raw output

```
nov,clean,predict_col_recall=0.604,predict_any=0.931,predict_step_acc=0.575,burst=0.3961,anomaly_score_dev=0.00e+00,segments=28321
nov,auc,type=random,burst_fraction=0.753,pred_error=0.753,unexpectedness=0.753,surprise_ewma=0.817,surprise_slow=0.687,context_mismatch=0.524,surprise_win20=0.623
nov,auc,type=skip,burst_fraction=0.687,pred_error=0.687,unexpectedness=0.687,surprise_ewma=0.745,surprise_slow=0.675,context_mismatch=0.539,surprise_win20=0.632
nov,auc,type=stutter,burst_fraction=0.567,pred_error=0.567,unexpectedness=0.567,surprise_ewma=0.572,surprise_slow=0.534,context_mismatch=0.512,surprise_win20=0.518
nov,auc,type=pooled,burst_fraction=0.666,pred_error=0.666,unexpectedness=0.666,surprise_ewma=0.711,surprise_slow=0.629,context_mismatch=0.525,surprise_win20=0.591
nov,auc_shuffled_control,burst_fraction=0.507,pred_error=0.507,unexpectedness=0.507,surprise_ewma=0.507,surprise_slow=0.515,context_mismatch=0.508,surprise_win20=0.509
nov,alpha_sweep,alpha=0.02:0.573,alpha=0.05:0.629,alpha=0.10:0.678,alpha=0.20:0.711,alpha=0.35:0.718,alpha=0.50:0.712,alpha=0.80:0.686
nov,best_signal=surprise_ewma,mean_auc=0.711
nov,threshold_at_fpr1=0.5495,tpr=0.047,signal=surprise_ewma
nov,threshold_fpr_actual=0.0098,clean_n=10779,anom_n=1221
nov,operating_point,target_fpr=0.01,threshold=0.5495,tpr=0.047,fpr=0.0098
nov,operating_point,target_fpr=0.05,threshold=0.5309,tpr=0.129,fpr=0.0496
nov,operating_point,target_fpr=0.10,threshold=0.5181,tpr=0.234,fpr=0.0992
nov,bins=clean:0.415,0.425,0.420,0.426,0.412,0.415,0.406,0.432,0.423,0.416;corrupt:0.609
nov,peak_rss=9MB
```