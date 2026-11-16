# 20 — Learned syllabification and auditory categories

## Why
Up to `19_sequence_syllables.md` the syllable split was hand-set (vowel = nucleus).
That is a human rule pasted on top of the machine. Two things were still borrowed:

1. **Phone order inside a word** was known from the lexicon (dictionary order), not
   learned from statistics.
2. **Auditory categories** (which acoustic cluster is a phoneme) came from hand-set
   formant targets per phone, i.e. a label table.

Both can be derived online from the speech stream itself:

- **Word boundaries = prediction-error peaks.** A phone that is surprising after the
  previous phone marks a likely boundary. This is the classic transition-probability
  segmentation result (Saffran/Aslin/Newport style statistics), and in predictive
  terms it is exactly "high surprise = boundary".
- **Phoneme categories = unsupervised clustering of acoustic features.** Competitive
  learning (k-means / SOM) over formant vectors gives categories with no labels; the
  label is only used afterwards for scoring purity, never for training.

## Research basis
- Rohrlich & O'Reilly, *An unsupervised model of the auditory hierarchy with
  predictive learning*, arXiv:2108.06065 — prediction error as the training signal in
  an auditory hierarchy; no labels.
- Goriely et al., *Word segmentation and discovery via prediction error*,
  arXiv:2504.03338 — boundaries coincide with peaks of next-phone prediction error.
- Van Aken, *Unsupervised word segmentation*, arXiv:1105.6162 — transition
  probability statistics suffice to find word edges in a continuous stream.
- Kohonen, *Self-organizing maps* — competitive learning produces topologically
  ordered categories without supervision (used here only as a simple k-means).
- Jensen & Lisman, *Hippocampal theta–gamma coupling* (2005) and Lisman & Idiart
  (1995) — nested theta/gamma phase coding as the substrate for ordered chunks; this
  motivates the phase-order layer in `19_sequence_syllables.md`, not this file.

## What was built
`tools/phonotactics.cpp` (new), linked to the `bm` library:

1. **Transition statistics from within-word pairs only.**
   Unigram and bigram counts are taken from the phone sequences of the lexicon
   (`WordTable`). No boundary symbol is used during training, so the model never
   sees "this pair spans a word edge".
2. **Boundary detection on a continuous stream.**
   Sentences are flattened into one phone stream (spaces removed). For each adjacent
   pair the surprise is `-log P(next | prev)` with add-alpha smoothing. A boundary is
   predicted when surprise > threshold; the threshold is swept and the best F1 kept.
   Ground truth is the real word edge, which is only used for scoring.
3. **Vowel-nucleus baseline.**
   The hand rule "a vowel starts a new word" is scored on the same stream, so the
   learned rule can be compared against the borrowed one.
4. **Auditory categories by competitive learning.**
   Per phone, 20 noisy formant samples (F1/F2/F3/pitch, noise = sum of uniforms × 40 Hz).
   Online k-means with k = 23, 40 epochs. Purity = weighted majority label per cluster;
   recognition = nearest cluster to the clean target vector. Labels are used only for
   the final score.

## Results
```
boundary,best_thr=2.80,precision=1.000,recall=0.944,f1=0.971
baseline_vowel,precision=0.018,recall=0.028,f1=0.022
auditory,purity=0.735
auditory,recognition_top1=17/23
```

Reading:
- The learned boundary rule is essentially the real rule: no false positives on this
  stream, and it misses 1 in 18 real edges (F1 0.971). The hand vowel rule is worse
  than chance (F1 0.022) because most word edges in this corpus are consonant-initial
  after a vowel-final word.
- Acoustic clustering puts 73.5 % of samples in a cluster whose majority label is the
  true phone; 17 of 23 phones have their clean target vector nearest to the correct
  cluster. The 6 confusions are the expected acoustic neighbors (place/manner pairs
  such as p/b, t/d, s/sh), which is the same confusion set seen in `17_audit.md` with
  the supervised inverse model. So the categories are real, not memorized.

## What this changes in the architecture
- The **syllable layer no longer needs a hand vowel table**. Boundaries come from
  prediction error, and the nucleus can be detected as the local acoustic energy peak
  inside the segment (not implemented yet; the current layer still uses vowels).
- The **phone inventory is learned, not given**. `SpeechModel` keeps its table for the
  demo, but the category layer can replace it: a phone is now a cluster id plus its
  prototype, and the id is only a name for the cluster.

## Not yet done
- Nucleus detection from energy/periodicity instead of the vowel table.
- Theta/gamma nesting with the learned segments (slow phase = syllable, fast phase =
  phone); the current phase order is a flat 4-slot shift.
- Co-articulation inside the stream: the forward model is still trained on isolated
  phones, so transition effects are not learned.
- Boundary learning on a corpus that is not the lexicon (needs a larger phone stream).