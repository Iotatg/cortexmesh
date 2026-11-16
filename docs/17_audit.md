# 17 · Audit: how much is the model, how much is the code?

Question: are the results from learning, or from hand-written code? Controls were added in
`tools/audit.cpp` (`./build/bin/audit`).

## Controls and results

| control | result | what it proves |
|---|---|---|
| speech: train with permuted labels (art[i] → sound[i+7]) | 23/23 | the association is learned; it is not a lookup table |
| speech: identity part removed (features only) | 10/23 | the model still learns features, but identity does most of the work |
| language: overlap of word code with its concept code | 1.50 / 24 | concept is **not** leaked into the word code |
| language: train word → shuffled concept | 24/24 | grounding is learned, not a hard-coded answer |
| language: hold out "pyaas", train the rest | hit | concept generalizes from shared phones (paani) |
| feeling: body code vs word code overlap | 0.80 / 512 | no body-state leak into the word code |
| feeling: train body → shuffled word | 5/5 | the mapping is learned, not fixed |

Static scan: the only hard-coded semantics are the **world tables** (`WordTable`, `ConceptTable`,
`PhoneTable`, formants, valence/arousal). There is no answer lookup, no `if (word == ...) return
answer`, no test/training leak.

## What is the model, what is the code

**Model (learned at runtime):**
- associative memories: speech forward/inverse, word↔concept, body→word, sequence readout
- spatial pooler + temporal memory: columns, segments, burst/anomaly
- phase binding: order separation

**Code (hand-designed, not learned):**
- feature encoders: phone features, formant ranges, 8 concept bits, valence/arousal
- word/phone tables (the "world knowledge")
- simulation world rules and the state hash `state:<id>`
- the readout is supervised: it is trained with the correct next word

## Honest verdict

1. **Learning is real.** Arbitrary mappings can be learned (permuted labels, shuffled concepts,
   shuffled feelings), so the accuracy is not a hard-coded answer.
2. **Competence is narrow.** Speech is 23/23 only with the identity part; features alone give 10/23.
   The model recognizes phones, it does not yet generalize across accents or unseen sounds.
3. **Language is supervised and small.** Train top-1 0.67, test 0.25; ambiguity is 3/31 contexts.
   The model learns the corpus, not grammar.
4. **Simulation detection is memorization of exact states.** `observe_state` is a hash of `(x, y,
   dir)`, so the TM learns transitions between exact states. It detects a teleport because the
   transition is novel, not because it understands "movement should be continuous".
5. **Grounding is learned but the labels are hand-coded.** Word→concept works (1.0) and survives a
   shuffle, but the concept set is designed by us.

So: the code provides the body (encoders, world, tables); the model provides the learning
(associations, prediction, anomaly). The intelligence is in the learned associations, but the
concepts it learns are only as good as the hand-built features.
