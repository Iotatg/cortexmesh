# 32 · Overnight run (13-14 Nov 2026)

Everything in this file was produced by running the tools in this repository, in one session,
with no manual tuning between runs. Raw logs are under `logs/`.

## Plan

| # | question | tool | status |
|---|---|---|---|
| 1 | does the word LM improve on 4x more text? | `streamlm` on wikitext-103 (86.5M tokens) | done, collapsed (D41) |
| 2 | is 3-word context better than 2-word? | `streamlm`, 2M-token bake-off | done |
| 3 | can a character model read code? | `charlm` on Python stdlib | done |
| 4 | does search fix greedy degeneration? | `charlm` beam search | done (no) |
| 5 | does cleanup re-ranking help decoding? | `streamlm` decode = 1 | done (no) |
| 6 | does cleanup denoise hold at scale? | `cleanup` | done |
| 7 | does a super-linear read-out add capacity? | `powermem` | done (no) |
| 8 | can the model retrieve a file from an issue? | `swebench` | done (chance) |

## Results

### Word LM (wikitext-103, 86.5M tokens, V = 5000)

| config | tokens seen | top-1 | top-3 | top-10 | MRR |
|---|---|---|---|---|---|
| wikitext-2, V=3000, 1 word ctx | 400k | 0.0637 | 0.0716 | 0.0878 | 0.0723 |
| wikitext-103, V=5000, 2 word ctx | 500k | 0.0696 | 0.0799 | 0.0938 | 0.0782 |
| wikitext-103, V=5000, 2 word ctx | 2M | 0.0589 | 0.0719 | 0.0904 | 0.0692 |
| wikitext-103, V=5000, 3 word ctx | 2M | 0.0630 | 0.0739 | 0.0927 | 0.0725 |

`inv_top1` (out-of-vocabulary target excluded) is the honest number, because 17 % of
targets are out of vocabulary at V = 5000 and the model can never rank them.

### Bake-off at a fixed 2M tokens (V = 5000, wikitext-103)

| config | top-1 | top-3 | top-10 | MRR | inv_top1 |
|---|---|---|---|---|---|
| 2-word ctx, xi=0.005 | 0.0589 | 0.0719 | 0.0904 | 0.0692 | 0.0710 |
| 3-word ctx, xi=0.005 | 0.0630 | 0.0739 | 0.0927 | 0.0725 | 0.0760 |
| 4-word ctx, xi=0.005 | 0.0652 | 0.0741 | 0.0916 | 0.0737 | 0.0787 |
| 3-word, V=10000 | 0.0608 | 0.0671 | 0.0865 | 0.0690 | 0.0681 |
| 3-word, subword=1 | 0.0630 | 0.0739 | 0.0927 | 0.0725 | 0.0760 |
| 3-word, xi=0.010 | 0.0554 | 0.0661 | 0.0843 | 0.0649 | 0.0669 |
| 3-word, xi=0.003 | 0.0653 | 0.0757 | 0.0934 | 0.0744 | 0.0788 |
| 3-word, xi=0.0025 | 0.0663 | 0.0763 | 0.0934 | 0.0751 | 0.0799 |
| 3-word, xi=0.0015 | 0.0685 | 0.0772 | 0.0924 | 0.0767 | 0.0827 |
| 3-word, xi=0.0008 | 0.0708 | 0.0772 | 0.0902 | 0.0779 | 0.0854 |
| 3-word, xi=0.0004 | 0.0726 | 0.0773 | 0.0896 | 0.0789 | 0.0876 |

Findings:

1. Context helps, with diminishing returns: 2 -> 3 words is +7 % in-vocabulary top-1, 3 -> 4 is
   +3.5 % for 33 % more compute. Three words is the sweet spot.
2. A larger vocabulary hurts: V=10000 loses 10 % of in-vocabulary top-1 at the same token
   count, because each word's code receives fewer Hebbian updates. Vocabulary should be sized to
   the data, not maximised.
3. The learning rate is the dominant knob and the trend is monotonic down to xi=0.0004 at 2M
   tokens (+23 % over xi=0.005). The optimum at 2M is not the optimum at 86M, so the full run
   uses xi=0.001.
4. `subword=1` changed nothing on this corpus (identical numbers): the flag only affects rare
   words, and at 2M tokens the sub-word path is not exercised.
5. Lower xi sharpens top-1 while slightly flattening top-10 (0.1115 -> 0.1081), i.e. the model
   becomes more confident and less diverse, which is the expected signature of slower, more
   selective weight growth.

### Character model on code (Python stdlib, 585k held-out chars)

| model | top-1 | top-5 | MRR | exact 16-char | mean prefix |
|---|---|---|---|---|---|
| ctx=2, N=256 | 0.3010 | 0.4824 | 0.3969 | — | — |
| ctx=8, N=256 | 0.3758 | 0.5966 | 0.4802 | 0.0010 | 0.725 |
| ctx=12, N=256 | 0.3821 | 0.6072 | 0.4896 | — | — |
| ctx=8, N=512 | 0.3919 | 0.6128 | 0.4981 | 0.0010 | 0.725 |
| ctx=8, N=512, beam 4 | 0.3919 | 0.6128 | 0.4981 | 0.0010 | 0.647 |
| ctx=8, N=512, xi=0.012 | 0.3405 | 0.5840 | 0.4529 | 0.0020 | 0.503 |
| unigram baseline | 0.3194 | — | — | — | — |

### Cleanup memory (D35, benchmark 40)

| codes | exact denoise | us/clean | RAM |
|---|---|---|---|
| 1k | 1.000 | 14.2 | 0.1 MB |
| 10k | 1.000 | 136.8 | 1.2 MB |
| 100k | 1.000 | 1469 | 12.2 MB |
| 500k | 1.000 | 9229 | 61.0 MB |

### Associative memory (D36, benchmark 39)

Linear read-out beats p=2 and p=3 at every load; shared-context case collapses (0.213 -> 0.053).

## The full-corpus run collapsed (D41)

| tokens seen | top-1 | inv_top1 |
|---|---|---|
| first half (43.2M) | 0.0065 | 0.0065 |
| second half (43.2M) | 0.0000 | 0.0000 |
| whole corpus (86.5M) | 0.0033 | 0.0039 |
| held-out valid | 0.0000 | 0.0000 |

Direct measurement at 8M tokens: mean weight 0.9902, 96.9 % of synapses above 0.9 at xi = 0.005.
At xi = 0.0005: mean 0.7797, 52.3 % above 0.9, in-vocabulary top-1 0.0808 (vs 0.0368).

## Negative results

| idea | why it failed |
|---|---|
| super-linear read-out | amplifies noisy strong synapses, discards weak consistent evidence |
| cleanup re-ranking in LM decode | snapping destroys the weighted vote the read-out depends on |
| beam search on char model | score is uncalibrated; beam prefers frequent characters |
| file retrieval from issue text | candidate paths carry no semantics; at chance without repo contents |

## What is still open

* forgetting proportional to activity, or per-context mean subtraction, so the matrix can absorb
  86M tokens without saturating (this is the blocking issue)
* representational fix for code continuation (indentation / block context), not search
* repo checkouts for a real SWE-bench retrieval baseline
* two-stage cleanup (sketch prefilter) for sub-linear 1M-code denoise