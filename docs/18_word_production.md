# 18 · Word-level production + co-articulation

## What was added

`SpeechModel::acoustic(phone, prev)` — a context-dependent acoustic code. The previous phone's
place of articulation pulls F2 toward it, so the same phone sounds different after different
neighbours (co-articulation).

`tools/wordspeech.cpp` — the word-level loop:

```
word code  ->  audio summary  (say)
audio summary  ->  word code  (hear)
```

Audio summary = OR of the phone acoustic codes (order is not encoded yet).

## Results (`./build/bin/wordspeech`)

```
words,covered=25
word_loop,hear_top1=21/25,hear_top3=24,say_top1=13/25,say_top3=19
word_heldout,pyaas_rank=4/25
coarticulation,phone=k,self_overlap=18,context_overlap_with_aa=4
coarticulation,phone=t,self_overlap=17,context_overlap_with_aa=2
coarticulation,phone=p,self_overlap=18,context_overlap_with_aa=2
context_inverse,top1=89/170
```

- **Hearing is strong**: audio → word 21/25, top-3 24/25.
- **Saying is weaker**: word → audio 13/25, top-3 19/25. The audio summary is a bag of phones;
  words that share phones (main/mujhe/madad) collide.
- **Held-out "pyaas" ranks 4/25** without ever being trained — it shares `p/aa/s` with `paani` and
  `pyaas`'s concept, so the model generalizes.
- **Co-articulation is real**: `k` overlaps itself 18 bits but only 4 bits with `kaa`; context
  changes the acoustic target.
- **Context-aware inverse** 89/170 (52%) — context helps, but prev+current codes overlap.

## Limits

1. **No temporal order in the audio summary.** OR is commutative; "paani" and "iinaap" would sound
   the same. Need a sequence (concatenate with silence gaps, or phase-bind positions).
2. **Saying is weak** because word codes are 24 bits and many words share phones. A hierarchical
   code (word → syllable → phone) would separate them.
3. **Context inverse is 51%** — the context code (prev art OR current art) is not very separable;
   bind with phase instead of OR.
4. **Formants are hand-set.** No auditory categories learned from exposure.

## Next

1. **Sequence audio**: concatenate phone codes with silence gaps (or theta-phase) and train the
   word→audio map on sequences.
2. **Syllable layer**: word → syllables → phones; learn syllable codes from co-occurrence.
3. **Learned auditory categories**: cluster heard sounds by prediction error.
4. **Phase-bind context** instead of OR for the context inverse.
