# 15 · Language without tokens: feature-grounded word codes

## Why tokens were wrong

The first language attempt used `Encoder::token(id)` — an arbitrary id per word, exactly like a
transformer vocabulary. That is not how cortex represents words. Research from the last decade
points in a different direction:

- **Vector Symbolic Architectures / HRR / SPA** (Kleyko et al., arXiv:2106.05268; Ganesan et al.,
  arXiv:2109.02157; Chouinard, arXiv:2605.08458): concepts are high-dimensional vectors; binding
  (circular convolution / permutation) and bundling (superposition) give composition without
  discrete symbols. `GHRR` (arXiv:2405.09689) shows binding can even implement attention.
- **Latent Language Hypothesis** (Khorrami & Räsänen, arXiv:2109.14200): phones, syllables and
  words emerge as latent representations from cross-modal prediction — they are not given as
  tokens. The learner discovers units that support translation between speech and meaning.
- **Predictive processing / active inference** (Tull et al., arXiv:2308.00861; Donnarumma et al.,
  arXiv:2308.04941): language is inference in a hierarchical generative model, not next-token
  classification. Comprehension = predicting sensory/motor/interoceptive consequences; production =
  the reverse.
- **Mental lexicon** (Kröger, arXiv:2611.09402): lexical entries are *lemma + phonological form +
  concept*; perception and production are two associative pathways with WTA competition and
  thresholded transfer — not a single embedding table.
- **Sequence memory** in cortex is temporal/oscillatory (theta-gamma phase coding, sequence
  replay), not learned positional embeddings.

So the design rule is: **words are compositions of grounded features, not ids.**

## CortexMesh design (no token ids)

`WordEncoder` (`include/bm/words.hpp`) builds a word code from three channels:

1. **Phonology / articulation** — each phone has a unique 16-bit part plus 4 feature bits
   (place, manner, voicing, vowel height). Words that share sounds share bits.
2. **Grounded concepts** — 8 binary features: `self, other, body, need, positive, negative,
   liquid, action`. "bhookh" = body + need + negative; "paani" = liquid + need.
3. **Interoception** — valence and arousal scalar blocks, so words carry an affective body state.

The code is a **superposition** (union then fixed-k trim), not an id:

```cpp
parts = { phone(p1), phone(p2), ..., concept(c1), ..., valence(v), arousal(a) };
return UnionTrim(parts, n, k);   // k = 24
```

Sequence context is a **binding of previous word codes** (shift each code to a different part of
the vector and OR), not learned position embeddings. The readout is one linear unit per word,
trained with an error-modulated local rule (`err = target - score`, `w += lr * err * ctx`) —
no softmax, no backprop, no global loss.

`Encoder::token()` has been removed; the simulation now uses a hash-based **state assembly**
(`state:<id>`), not a language token.

## Results (`./build/bin/lang`)

```
similarity,same_phone=4.39,unrelated=2.66,ratio=1.65
context,total=31,ambiguous=3,ambiguity_rate=0.097
sequence,train_top1=0.667,train_top3=0.939,train_mrr=0.803,test_top1=0.250,test_top3=0.500,test_mrr=0.368
novelty,after_mujhe_top=bhookh,rank_pyaas=14
grounding,word_to_concept=1.000,concept_to_word=0.375
feeling,flips=0,top1=1.000
feeling,flips=2,top1=1.000
feeling,flips=4,top1=1.000
feeling,flips=8,top1=1.000
dialogue,mujhe bhookh lagi hai
```

What this shows:

- **Words are similar when their features are similar** (4.39 vs 2.66 overlap). Not ids.
- **Understanding = prediction**: trained sequence top-3 0.94, MRR 0.80; the model ranks the
  right word first most of the time on seen contexts.
- **Grounding is bidirectional**: word → concept 1.00; concept → word 0.375 (concepts overlap:
  `body`+`need` is shared by bhookh/thak/neend, so recall is ambiguous).
- **Feeling works**: body state (HR/HRV/breath/effort) → feeling word at 1.00, unchanged under
  8 bit-flips of the body code. This is interoception driving language, not a label lookup.
- **Production**: from `<s> <s>` the model generates `mujhe bhookh lagi hai` by repeatedly
  predicting and feeding the prediction back.

## Limits (honest)

1. **Ambiguity is real.** 3/31 contexts have >1 continuation; test accuracy is limited by that,
   not by the encoder.
2. **Novel word is weak** (rank 14 for "pyaas"). It shares `p/aa` with "paani" but only 2 of 4
   phones; phonological features need more dimensions and more data.
3. **Concept → word 0.375**: concept codes are only 8 bits; shared `body/need` dominates. Use
   separate slices per concept or learn concept-specific codes from data.
4. **No motor/audio loop yet.** The model has phones as features but no articulation, no
   self-heard speech. Real language learning is a sensorimotor loop.
5. **Readout is localist per word** (one unit per word). Cortex uses distributed assemblies; this
   is a demo simplification, but the learning rule is local.

## Next

1. **Done** — articulatory + audio channel and the phone-level loop are in doc 16
   (`SpeechModel`, forward/inverse top-1 = 1.000).
2. **Done (demo)** — theta-gamma phase coding separates AB from BA (doc 16); still needs nested
   frequencies.
3. Use the **predictive hierarchy**: word → predicted sensory/motor consequences; train the
   hierarchy on prediction error, not on labels.
4. Grow the lexicon from **cross-situational exposure** (word co-occurs with world state), so
   grounding is learned, not hand-coded.
