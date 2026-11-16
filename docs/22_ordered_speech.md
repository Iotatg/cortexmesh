# 22 · Ordered speech: nested theta-gamma, co-articulation, slow-phase codes

## Why
`19_sequence_syllables.md` added order with a flat 4-slot shift and a hand syllable
split. Two things were still missing:

1. **Nested phase coding** (Lisman & Idiart 1995; Jensen & Lisman 2005): a slow theta
   phase for the syllable, a fast gamma phase for the phone inside it. A flat shift
   throws away the hierarchy.
2. **Co-articulation in the code**, not only in the acoustic model: the phone code used
   for a sequence should depend on its neighbour.

## What was built
`tools/sequence.cpp`:

- `SpeechModel` supplies 1024-bit phone codes (`acoustic(phone, prev)` for
  co-articulated context, `acoustic(phone)` for isolated).
- Three order modes:
  - `bag`: OR of phone codes, no phase.
  - `flat4`: rotate phone i by `(i mod 4) * n/8`.
  - `nested`: rotate by `(syll mod 2) * n/2 + (i mod 4) * n/16`.
- Rotation (not shift) keeps every code inside the same 1024-bit space and preserves
  the sparse code exactly; trimming to k=24 is done by taking the first k set bits.
- Syllables come from the vowel-nucleus rule (still hand-set; learned nuclei are the
  next step).
- Two associative memories: `audio → word` (hear) and `word → audio` (say), both
  trained on 24 of 25 vocabulary words; `pyaas` is held out.

Metrics:
- hear/say: rank of the first target bit (top-1, top-3).
- `sayov`: mean overlap between produced and target audio.
- `rev`: overlap between a word and its phone-reversed version (1.0 = order-blind).
- `ctx`: overlap between co-articulated and isolated audio (1.0 = context-blind).
- `slowphase`: mean overlap of the low half of the code (slow phase) for the *same*
  syllable across words vs *different* syllables.

## Results
```
bag     coart=0 hear=18/24(1.000) say=15/24(1.000) sayov=0.865 rev=0.993 ctx=0.993
bag     coart=1 hear=18/24(1.000) say=15/24(1.000) sayov=0.865 rev=0.976 ctx=0.983
flat4   coart=0 hear=22/24(1.000) say=18/24(1.000) sayov=0.889 rev=0.326 ctx=0.993
flat4   coart=1 hear=22/24(1.000) say=18/24(1.000) sayov=0.889 rev=0.326 ctx=0.988
nested  coart=0 hear=21/24(1.000) say=18/24(1.000) sayov=0.899 rev=0.309 ctx=0.993
nested  coart=1 hear=21/24(1.000) say=19/24(1.000) sayov=0.901 rev=0.309 ctx=0.908
heldout,pyaas,rank=0,of=24
slowphase,same=0.283,diff=0.151,n_same=50,n_diff=896
```

Reading:
1. **Order is explicit**: bag has reversal overlap 0.99 (it cannot tell a word from
   its reversal); both phase modes drop it to ~0.31. Phase binding carries order.
2. **Nested is not better than flat here** (21 vs 22 hear, same say). With 2 slow slots
   and 4 fast slots the two codes are nearly the same amount of information; the
   advantage should appear with longer words and more syllables.
3. **Co-articulation changes the code** only in nested mode (ctx 0.993 → 0.908). In bag
   and flat the identity block dominates, so context is nearly invisible — a reminder
   that the *encoding* decides what the network can use, not the acoustic model alone.
4. **Held-out `pyaas` ranks 1st** (rank 0) among 24 trained words, using only shared
   phone codes. Generalisation by composition, not memorisation.
5. **Slow phase carries syllable identity weakly but consistently**: 0.283 for same
   syllable vs 0.151 for different (1.87x). The nested code does put syllable
   information in the slow half, but it is diluted by the fast rotations that spill into
   it.

## Interpretation
Order can be represented without tokens or attention: a sparse code per phone, rotated
into a phase slot, OR-ed into a fixed-size assembly. The reversal test is the honest
control — a bag-of-phones model passes recognition but fails reversal, which is exactly
the failure mode of a transformer-free bag model.

The weak point is the *amount* of order: 4 slots for phones, 2 for syllables. Real
theta-gamma nesting uses ~7 gamma cycles per theta cycle, and the slow phase should be
a continuous position code, not 2 slots. Also, rotation by a fixed stride makes codes
that are 1 slot apart differ in only a few bits, so the model can confuse neighbours;
a random permutation per slot would be sharper.

## Not yet done
- 7:1 theta-gamma ratio and a continuous slow-phase position code.
- Random per-slot permutations instead of a fixed stride.
- Learned syllable boundaries feeding the phase assignment (see `20_phonotactics.md`).
- Transition-trained forward model (co-articulation inside the sequence, not just the
  pairwise context vector).
- Longer words: measure whether nested beats flat at 3+ syllables.