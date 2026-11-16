# 31 · Code as a first-class stream (character-level)

Real code is a different distribution from prose: indentation is meaningful, identifiers are
long, punctuation is dense, and the alphabet is 95 printable ASCII characters instead of ~30.
The character model (`tools/charlm.cpp`) is the right instrument because it has no vocabulary
and therefore no OOV by construction.

## Setup

| item | value |
|---|---|
| train corpus | 573 Python stdlib files (`/usr/lib/python3.11`, < 60 KB each), 6.6 MB |
| validation | 56 files held out by file hash, 571 KB, 585,471 characters |
| alphabet | 95 printable ASCII (code mode keeps case, folds whitespace to space) |
| code width | N = 256 bits, K = 8 active (3.1 %) |
| context | last `ctx` characters, each in its own N-bit block (positional) |
| learning | one pass, EMA Hebbian, xi = 0.004 unless stated |

## Context length is the lever

| ctx (chars) | top-1 | top-5 | MRR |
|---|---|---|---|
| 2 | 0.3010 | 0.4824 | 0.3969 |
| 4 | 0.3525 | 0.5526 | 0.4485 |
| 6 | 0.3728 | 0.5847 | 0.4727 |
| 8 | 0.3758 | 0.5966 | 0.4802 |
| 12 | 0.3821 | 0.6072 | 0.4896 |
| unigram baseline | 0.3194 | — | — |
| previous-char baseline | 0.2281 | — | — |

Findings:

1. At ctx = 2 the model is *below* the unigram baseline (0.3010 vs 0.3194): two characters of
   context are not enough to beat "always predict the most frequent character". The model needs
   more context before it adds anything.
2. The gain from context is steep to ctx = 6 and then flat (+0.009 from 6 to 12 for 2x the
   cost). The useful context for code is short-range: indentation, the previous token, and the
   line being typed.
3. The previous-character baseline (0.2281) is far below the model at every context length, so
   the model is genuinely using the context block, not just the last character.
4. Every character is in the alphabet by construction (oov = 0.000), which is what makes this
   usable on identifiers the model has never seen.