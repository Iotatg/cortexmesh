# 33 · Summary (night of 13-14 Nov 2026)

## What changed in the repository

| area | change |
|---|---|
| `tools/powermem.cpp` | read-out power sweep for the associative matrix (D36) |
| `tools/swebench.cpp` | file retrieval from real GitHub issue text |
| `tools/prep_data.py` | parquet -> text conversion for wikitext-103, gsm8k, SWE-bench Lite |
| `tools/charlm.cpp` | code mode (95-char alphabet, case preserved), runtime context width, runtime code width, greedy continuation benchmark, beam search |
| `tools/streamlm.cpp` | multi-epoch training, cleanup decode mode, decode baseline for comparison |
| `tools/bench_suite.sh` | one command runs every benchmark into `logs/suite_<tag>/` |
| `data/` | wikitext-103 (86.5M tokens), Python stdlib (573 files), SWE-bench Lite (300 instances) |
| `docs/31`, `docs/32`, `docs/33` | code model, overnight report, this summary |

Tests: 20/20 pass (`ctest`), zero warnings in the default build.

## Measured results

| question | answer |
|---|---|
| does more text help? | no: the full-corpus run collapses (D41) because the Hebbian matrix saturates |
| does more context help? | 2 -> 3 words +7 %, 3 -> 4 words +3.5 % (2M-token bake-off) |
| does a bigger vocabulary help? | no, V=10000 loses 10 % vs V=5000 |
| which learning rate? | lower is better at fixed tokens: xi=0.005 -> 0.0004 is +23 % in-vocabulary top-1 at 2M |
| can the model read code? | yes at character level: 0.392 next-char top-1, 8x the word-level model |
| can it write code? | not yet: exact 16-char continuation 0.001, mean correct prefix 0.73 chars |
| does search fix writing? | no, beam search is slightly worse (D39) |
| does cleanup help decoding? | no, it hurts (D37); it helps discrete codes (hops 0.00 -> 0.99) |
| does cleanup scale? | accuracy flat at 1.000 to 500k codes, 61 MB, 9.2 ms per clean |
| does a super-linear read-out add capacity? | no, worse at every load (D36) |
| can it locate a file from an issue? | no, at chance: path-only candidates carry no semantics |

## Honest gap to a frontier model

The benchmarks published for Gemini 4 Hydrogen (SWE-sweep 13.5 %, Terminal-Bench 57.0 %, HLE
58.8 %) measure multi-step tool use, repository-scale editing, and expert-level question
answering. This project measures next-symbol prediction, sequence memory, and associative
retrieval with fixed-size state and no backpropagation. The two are not on the same axis, and
nothing produced tonight closes that gap.

What the night does establish is the part that is measurable here: the architecture learns
(hold-out top-1 rises with context and with a lower learning rate), retains what it learned in the
short term (retention 1.000 after a second list), retrieves without a context window (age curve
flat to 1024 steps), denoises at scale (1.000 exact to 500k codes), and stays inside 65 MB for an
86M-token model.

It also found the blocking defect: without forgetting, more data makes the model worse. The
86.5M-token run ends at chance (D41), and the same configuration is at 0.076 in-vocabulary top-1
after 2M tokens. Any claim about scale for this architecture has to be made after that is fixed.

## Next experiments, in order

1. forgetting: activity-proportional decay, or per-context mean subtraction, so 86M tokens do not saturate
   the matrix; re-run the full corpus with the fix and with xi=0.001
2. represent the line (indentation, block, bracket stack) for code, not just the last 8 chars
3. two-stage cleanup: sketch prefilter, then exact overlap, to make 1M-code cleanup sub-linear
4. repo checkouts for a real SWE-bench retrieval baseline
5. hierarchy: region over regions, so context can be longer without a bigger matrix