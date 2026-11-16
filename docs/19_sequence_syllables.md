# 19 · Ordered audio + syllable layer

## What was added

1. **Phase-ordered audio** — each phone's acoustic code is shifted by its position (`i mod 4`),
   so "paani" and "iinaap" are different vectors. This is the theta-gamma idea at the audio level.
2. **Syllable layer** — phones are split at vowel nuclei into syllables (onset + vowel + coda);
   syllables are phase-bound into a word audio summary.

## Results (`./build/bin/wordspeech`)

```
word_loop,hear_top1=21/25,hear_top3=24,say_top1=13/25,say_top3=19
sequence,hear_top1=24/25,hear_top3=25,say_top1=18,say_top3=21,reversal_overlap=24.6
syllable,words=25,syllables=39,hear_top1=24,say_top1=21
coarticulation,phone=k,self_overlap=18,context_overlap_with_aa=4
context_inverse,top1=89/170
```

- **Order matters now**: reversed words overlap only 24.6 bits with the forward audio (a bag-of-phones
  would be much higher).
- **Syllables help production**: say top-1 13 → 18 (phone level) → 21 (syllable level).
- **Hearing is near-perfect**: 24/25 at both levels.
- **Co-articulation** still changes the target (18 → 4 overlap).

## Limits

1. **Syllabification is hand-written** (split at vowels). It is not learned from exposure.
2. **Only 4 phases**; real theta-gamma is nested (slow phase for syllable, fast for phone).
3. **Say is still not perfect** (21/25): words that share most phones (main/madad/mujhe) remain
   close.
4. **No auditory categories** learned from sound; formants are still hand-set.

## Next

1. **Learned syllabification**: discover syllable boundaries from transition statistics (which phone
   pairs are predictable).
2. **Nested theta-gamma**: slow phase for syllable index, fast phase for phone index.
3. **Auditory categories**: cluster heard sounds by prediction error, then map categories to phones.
4. **Co-articulation inside sequences**: train the forward model on real transitions, not isolated
   phones.
