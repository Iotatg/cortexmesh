# brain

An intelligence system built from neuroscience first principles: **no transformers, no
attention, no backpropagation, no gradient descent on a global loss**.

Learning is local (Hebbian + eligibility + broadcast neuromodulation), representations are
sparse (1–4 % active), and prediction is the primary signal.

## Status (phase 0 complete)

- `bm` library: `SDR`, `Encoder`, `ScalarEncoder`, `SpatialPooler`, `TemporalMemory`, `Region`,
  `ModulatorSystem`, `Salience`, `Valuation`, `Hippocampus`, `Interoception`, `DualController`.
- Tests: 30 checks, all passing (`ctest`).
- CLI: `brain_cli encode <text>`, `brain_cli seq`.
- Docs: `docs/00_index.md` → `docs/13_discoveries.md` (including measured discoveries D1–D8).

Measured: region learns a 32-item sequence and predicts it with **94–100 % accuracy**, and
flags unseen transitions as anomalies (`docs/11_benchmarks.md`). Focus follows an
inverted-U: moderate gain learns fastest, extreme gain degrades (`test_core`).

Phase 1.5: neuromodulators (dopamine RPE, noradrenaline novelty, acetylcholine
precision/focus), salience gain, interoception, valuation, hippocampus, dual controller.

Measured discoveries (`docs/13_discoveries.md`):
- capacity law `columns >= 32 x items`;
- winner boosting destroys code stability (removed);
- long eligibility traces destabilise value learning (short traces only);
- forward replay consolidates, reverse/random do not;
- burst rate is a calibrated anomaly signal;
- per-action value heads beat conjunction readouts (gap 1.0 vs 0.395);
- focus has an inverted-U (optimum near 1.0).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
cd build && ctest --output-on-failure
```

## Run

```bash
./build/bin/brain_cli encode hello
./build/bin/brain_cli seq
```

## Experiments

```bash
# Language: feature-grounded word codes, no token ids
./build/bin/lang

# Sensorimotor speech loop + theta-gamma order coding
./build/bin/speech

# Word-level production, order + syllables
./build/bin/wordspeech

# Audit: is it the model or the code? (controls)
./build/bin/audit
./build/bin/phonotactics
./build/bin/curriculum
./build/bin/sequence
./build/bin/streamlm data/wikitext2.txt 5000 0.002

# Streaming LM with generation (corpus vocab xi norm max_tokens gate eval subword ctx2 gen seed temp npos)
./build/bin/streamlm data/tinystories_valid.txt 2000 0.002 1 300000 0 "" 0 0 30 once 1.0 0

./build/bin/bench

# New modules: character LM, SDM memory, TM novelty ROC
./build/bin/charlm data/wikitext2_head.txt 0 0.002 -1 data/tinystories_head.txt 200 the
./build/bin/sdm
./build/bin/novelty

# Context-free retrieval: old info without a fixed window
./build/bin/contextfree

# Cleanup memory: denoise a retrieved code before re-using it
./build/bin/cleanup

./build/bin/experiments capacity
./build/bin/experiments eligibility
./build/bin/experiments replay
./build/bin/experiments anomaly
./build/bin/experiments action
./build/bin/experiments focus
```

## Layout

```
include/bm/      public headers
src/             implementation
tests/           test_core (assert-based, no external deps)
tools/           experiments and measurement tools (see table)
docs/            research, architecture, learning rules, roadmap, API, benchmarks
data/            corpora (wikitext-2, wikitext-103, TinyStories, Python stdlib, SWE-bench Lite)
logs/            raw output of every benchmark run
```

## Measurement tools

| tool | what it measures |
|---|---|
| `sim` | anomaly detection / rule violation in a simulated agent |
| `curriculum` | belief revision, hop depth, shuffled controls |
| `sequence` | temporal-memory sequence capacity |
| `textmodel` | text prediction with a plain associative memory |
| `streamlm` | word-level streaming LM (vocabulary, context width, decode mode, epochs) |
| `charlm` | character-level LM; code mode, context sweep, greedy and beam continuation |
| `sdm` | Kanerva sparse distributed memory vs dense associative memory |
| `novelty` | temporal-memory novelty, ROC and AUC |
| `contextfree` | retrieval without a fixed context window |
| `cleanup` | cleanup memory: denoise, hops, latency, scale |
| `powermem` | read-out power sweep (capacity of the associative matrix) |
| `swebench` | file retrieval from real GitHub issue text |
| `wordspeech`, `phonotactics`, `audit` | language primitives |
| `bench` | raw associative-memory speed and size |

Run all of them with `tools/bench_suite.sh <tag>` (writes `logs/suite_<tag>/`).

## Design rules

1. Local learning only.
2. Sparse codes only.
3. Prediction first, behaviour second.
4. Online, single pass.
5. Fixed-size state; memory lives in synapses.

## Next

`docs/05_roadmap.md` phase 1: temporal pooling, variable-order sequences, anomaly ROC,
homeostatic sweep, then hierarchy with feedback.