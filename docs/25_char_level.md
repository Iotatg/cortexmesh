# 25 · Character-level streaming language model

`tools/charlm.cpp` is the character-level counterpart of `tools/streamlm.cpp`. It uses the
same local mechanism: a sparse context SDR addresses a weight matrix, the matrix is read
out through a sparse inverted index, and learning is a single-pass EMA Hebbian update with
predict-before-learn. There is no softmax in learning, no backprop, no optimizer, and no
global loss. The only softmax is in the generation sampler.

The point of the experiment is D27: the word-level model has 22.5% OOV on TinyStories, so
its headline score was partly a vote for UNK. A character output space is closed, so OOV is
zero by construction and every prediction is a real symbol.

## Design

| item | value |
|---|---|
| alphabet | 48 symbols: `a`-`z`, `0`-`9`, space, `. , ; : ! ? ' " ( ) -` |
| alphabet construction | scanned from the training corpus, lower-cased, whitespace folded to space |
| context | last `k = 2` characters |
| context encoding | positional concatenation: position `p` owns bits `[p*N, (p+1)*N)` |
| code size | `N = 256`, `K = 8` active bits (3.1% dense) |
| context width | `k*N = 512` bits, 16 active |
| readout | top-64 output-bit scores from the N-bit accumulator, inverted index to characters |
| weights | `(k*N) x N` floats = 131,072 floats = 0.5 MB |
| learning | `row[j] += xi * (1 - row[j])` on the target's active bits, `xi = 0.002` |
| frequency normalisation | implemented but off (`kNormalize = false`); D24 measured no gain |
| passes | one, online, no stored pairs |

Positional concatenation was used, not OR-binding. OR-binding was shown to lose order
(D23/D25), and here each character position gets a disjoint 256-bit block, so `ab` and `ba`
activate different context bits. This is reported in the `encoding=positional_concat` field.

The character alphabet is closed by construction: characters outside the allowed set are
filtered out before prediction. This is why `oov=0.000`; it is not a claim that every byte
in the corpus is covered. The number filtered is reported as `skipped` (train) and
`eval_skipped` (eval).

## Commands

```bash
cd /workspace/brain
cmake -S . -B build_char -DCMAKE_BUILD_TYPE=Release
cmake --build build_char -j4 --target charlm
./build_char/bin/charlm /workspace/brain/data/wikitext2_head.txt 0 0.002 -1 /workspace/brain/data/tinystories_head.txt
```

Generation (trains on 300k characters, then samples 200 characters at temperature 0.8):

```bash
./build_char/bin/charlm data/wikitext2_head.txt 0 0.002 300000 data/tinystories_head.txt 200 the
```

Smoke shape used by ctest:

```bash
./build_char/bin/charlm data/wikitext2_head.txt 300
```

The second CLI argument is the word-model vocabulary size; the character model accepts it
and ignores it (`vocab_or_unused`).

## Measured results

Full run: train on the first 1,734,789 filtered characters of `wikitext2_head.txt`
(1,750,020 bytes, 15,231 bytes skipped), then evaluate without learning on 1,988,663
filtered characters of `tinystories_head.txt` (11,337 skipped). Wall time 46-49 s
across repeated runs.

| set | top-1 | top-5 | MRR | n_chars | alphabet | OOV |
|---|---:|---:|---:|---:|---:|---:|
| train, all | 0.3380 | 0.6533 | 0.4782 | 1,734,789 | 48 | 0.000 |
| train, first half | 0.2934 | - | 0.4508 | 867,394 | 48 | 0.000 |
| train, second half | 0.3826 | - | 0.5056 | 867,395 | 48 | 0.000 |
| eval, TinyStories | 0.4384 | 0.6222 | 0.5334 | 1,988,663 | 48 | 0.000 |

Baselines on the eval set:

| baseline | accuracy |
|---|---:|
| previous character (`char,prev_baseline`) | 0.0200 |
| most frequent training character, space (`char,unigram_baseline`) | 0.1966 |

Resource use:

| metric | value |
|---|---:|
| train seconds | 22.5-24.0 |
| peak RSS, train | 7 MB |
| peak RSS, whole run | 11 MB |
| weight matrix | 0.5 MB |
| wall time, whole run | 46-49 s |

The first-half/second-half split shows learning continuing rather than collapsing:
0.2934 -> 0.3826. This is consistent with retention, but it is not a controlled retention
test: the second half contains different articles, so easier text is a confound.

## Comparison with the word-level model

The word-level numbers are from D24 and D27 (283k tokens, known-word ranking where D27
applies). The units differ: the word model predicts a whole word, the character model
predicts one character. The comparison below is therefore about coverage and transfer, not
a claim that 0.438 character accuracy is "better than" 0.091 word accuracy in the same
unit.

| metric | word-level (D24/D27) | character-level (this work) |
|---|---:|---:|
| OOV on TinyStories | 22.5% | 0.000 by construction |
| same-domain top-1 | 0.105 inv (D27), 0.160 all (D24) | 0.3380 train |
| transfer top-1 | 0.091 inv | 0.4384 eval |
| transfer top-5 | not reported | 0.6222 eval |
| transfer top-10 | 0.151 inv (D27) | not reported |
| transfer MRR | 0.093 inv | 0.5334 eval |
| peak RSS | 15 MB | 11 MB |
| prediction unit | word | character |

What improved:

1. OOV disappears. Every scored target is in the 48-symbol alphabet, so the score cannot be
   inflated by predicting UNK. The word model's 22.5% OOV on TinyStories is the failure this
   was meant to remove.
2. Transfer is strong. The same weights, with no retraining and no learning on the eval
   corpus, score 0.4384 top-1 on TinyStories. The word model's in-vocabulary transfer was
   0.091 top-1 (D27).
3. The model is small: a 0.5 MB weight matrix, 7 MB peak RSS during training, and 11 MB
   for the whole run (including the eval corpus) at 1.73M training characters.
4. Learning does not collapse over the run: first-half 0.2934, second-half 0.3826.

## What did not improve

| claim | measurement |
|---|---|
| "Longer context helps" | false here: k=1 0.2816, k=2 0.2833, k=3 0.2502, k=4 0.2312 (283k train top-1) |
| "Character generation is sentence-like" | false: see below |
| "Top-5 is close to solved" | no: 0.6222 on TinyStories, 0.6533 on train |
| "OOV=0 means full byte coverage" | no: 11,337 eval chars (0.57%) were filtered out, not predicted |

Context and code sweep, 283k training characters, train top-1:

| k | N | K | train top-1 | train seconds |
|---:|---:|---:|---:|---:|
| 1 | 256 | 8 | 0.2816 | 4.56 |
| 2 | 256 | 8 | 0.2833 | 3.86 |
| 3 | 256 | 8 | 0.2502 | 4.11 |
| 4 | 256 | 8 | 0.2312 | 5.34 |
| 2 | 128 | 4 | 0.2687 | 3.80 |
| 2 | 512 | 16 | 0.2826 | 7.30 |
| 3 | 128 | 4 | 0.2454 | 5.36 |
| 3 | 512 | 16 | 0.2487 | 8.40 |

Readout width has no measurable effect at N=256 (283k train, then eval):

| top bits | train top-1 | eval top-1 |
|---:|---:|---:|
| 32 | 0.2840 | 0.3138 |
| 64 | 0.2833 | 0.3134 |
| 128 | 0.2833 | 0.3134 |
| 256 | 0.2833 | 0.3134 |

## Generation

The required sample, after 300k training characters, seed `the`, temperature 0.8, 200
sampled characters (the seed is printed, then 200 characters):

```text
char,generate=thewarkazdtor. atyt"rmec5eisyeetopiyzgq2ty atranamol8 (se elar si : c3o8qri thzaiolotss ang art eatind 'etich rony eenze9i3c'enettraic6or aio s ( . gettsf9ratepwseaoscn7iedeez9 dent6t tanisy a s;7a:itora
```

After training on the whole file, the same sampler produces:

```text
char,generate=the5  ciiaxasurtiduiakey1oasqunsib yw(z!rpossueildsc9 son scts iogx n2r)f osio6cerdunir e careerostiey ; siby a sa loo"ubr3cutanspiagu2isitso st: beyooopgfeariq"tit"aw8homuspbgebltsmunalarsq iss;de5  ell
```

The word-level samples in D29 were a bag of words:
`"once sick him mighty walking never to house berries farm ... pony little brave"`.
The character model is not a bag of words and not a sentence either: it is sub-lexical.
It produces local spelling fragments (`the`, `co`, `st`, `ent`, `tion`, `ing`), spaces in
plausible places, and then random letters and digits. It has no syntax.

A diagnostic truncation to the top-6 characters before sampling (not the shipped rule)
gives a more word-like sample, which shows the model does contain local spelling knowledge:

```text
thes a ie in an terations ont en a dintarit a erestee an t ere t ra ds ea en tore se tio te tha t or s ere se t of r as ini
```

It is still not a sentence. The shipped tool keeps the specified top-64-bit readout and
temperature 0.8; the truncation is reported only as a diagnostic.

## Honest failure and limitations

1. **Generation is the failure.** The character model fixes coverage, not grammar. At
   temperature 0.8 the sample is a character salad with local fragments; lowering the
   temperature or truncating the tail makes it more word-like but never syntactic. This is
   the same linear-readout limit as D29: context bits are associated with next-character
   bits, which learns local co-occurrence, not phrase structure.
2. **Longer context hurts.** k=2 is the best context here; k=3 and k=4 are worse
   (0.2502 and 0.2312 vs 0.2833). Adding positions adds more context bits than the EMA can
   fit from one pass, so each row is trained on fewer examples and the readout gets noisier.
   This is consistent with D29 finding 3 (a 4-word positional context gave no gain over a
   bigram), but it means the "last k characters" idea does not scale in this design without
   a different readout or a slower second stage.
3. **OOV=0 is partly definitional.** 11,337 TinyStories characters (0.57%) were outside the
   training alphabet and were skipped. The model never scores them. A byte-level model or a
   larger alphabet would be needed to claim literal full coverage.
4. **Scores are saturated.** Context rows approach 1.0, so char scores are sums of many
   near-1 weights (top scores around 100-128 with a small margin). Ranking works, but the
   small margin is why generation is noisy and why a sharper sampler changes the output so
   much.
5. **The first/second-half gain is confounded.** The second half is different text, not a
   held-out replay of the first half. It is evidence that learning continues, not a clean
   retention measurement.
6. **Accuracy is per character.** 0.4384 character top-1 is not 0.4384 word top-1. A fair
   word-level comparison was not run; the defensible claims are OOV elimination, transfer
   without retraining, bounded memory, and the negative result on longer context.