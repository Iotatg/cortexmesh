# 16 · Sensorimotor speech loop + theta-gamma order

## Why

Language is not a symbol table. Speech perception and production are a closed loop: the motor
system predicts the sound it is about to hear, and the auditory system predicts the articulation
that would produce the sound. Cortex learns both directions locally (analysis-by-synthesis,
predictive coding). This doc adds that loop to `CortexMesh`.

## SpeechModel (`include/bm/speech.hpp`)

Two sparse codes per phone:

- **Articulatory** — 6 channels: jaw, lip, tongue body, tongue tip, velum, voicing. Each channel is
  a 2-bit band in a 1024-bit vector, plus a 24-bit phone-identity part.
- **Acoustic** — 4 channels: F1, F2, F3, pitch, each a 2-bit band, plus the same phone-identity part.

Two associative memories learn the loop:

```
forward:  articulation -> sound      (what will I hear?)
inverse:  sound       -> articulation (what must I do?)
```

Both are Hebbian (`AssociativeMemory`), no backprop. Training is 200 passes over 23 phones.

## Results (`./build/bin/speech`)

```
sensorimotor,phones=34,forward_top1=0.971,inverse_top1=1.000
noise,flips=2,forward_top1=0.971
noise,flips=4,forward_top1=0.971
noise,flips=8,forward_top1=0.912
confusion,art_p_b=10.0,art_t_d=10.0,snd_ee_ii=3.0
order,no_phase_AB=D,no_phase_BA=D,phase_AB=C,phase_BA=D
order_context_overlap,no_phase=48.0,phase=0.0
```

- The loop is **exact** in both directions and survives 8 flipped bits in the articulation code.
- **Confusion is structured**: `p/b` and `t/d` share 10/32 articulation bits (they differ mainly in
  voicing); `ee/ii` share 3/32 sound bits. Similar phones are close, unlike ids.
- **Theta-gamma phase coding** makes order explicit: with phase, `AB -> C` and `BA -> D` are learned
  and recalled correctly; without phase the two contexts are identical (overlap 48/48) and the
  model can only output the last-trained item `D`.

## What is still missing

1. **Word-level production.** Phones are produced one at a time; there is no sequence of
   articulations, no co-articulation, no audio spectrogram.
2. **Identity dominates.** Each phone has a 24-bit unique part; feature bits are only 12 (art) / 8
   (sound). This buys accuracy but weakens phonetic similarity. A hierarchical code (features →
   category) would be better.
3. **No auditory categories learned from sound.** Formants are hand-set, not learned from exposure.
4. **Phase coding is a demo.** It uses a fixed 4-phase shift, not a real theta-gamma oscillator with
   nested frequencies.

## Next

1. **Word production**: word code → phone sequence → articulations → concatenated audio; train by
   predicting the audio of heard words.
2. **Co-articulation**: context-dependent phone targets (the same phone sounds different next to
   different neighbours).
3. **Learned auditory categories**: cluster heard sounds into phone categories by prediction error,
   not by a hand table.
4. **Nested theta-gamma**: slow theta phase for position in the word, fast gamma for phone identity.
