# 21 · Knowledge induction: facts, multi-hop, interference, replay

## Why
Language so far predicted the next word from context. That is syntax-level competence,
not knowledge. A PhD-level system has to hold *facts* (rose is a plant, plant needs
water) and combine them (rose needs water), and it has to keep learning new facts
without destroying old ones.

All of it must stay local: one Hebbian matrix per relation, SDR in / SDR out, no global
loss.

## What was built
`tools/curriculum.cpp`:

- **Entities** are SDRs, in two variants:
  - `hash`: `Encoder::symbol("ent:name")` — random, near-orthogonal (control).
  - `feature-bundle`: `Encoder::bundle` of feature codes (`living`, `plant`, `soft`,
    `sweet_smell`, …) plus a private name bit-block. Entities that share features
    share bits, which is what makes generalisation and interference possible.
- **Relations** are separate `AssociativeMemory(512, 512, k=24, lr=0.05)`:
  `isa` (entity→category), `has` (entity→attribute), `lives` (entity→place),
  `needs` (category→thing).
- **Sets** A (gulab, machli, paani), B (kutta, billi, roti), C (ped, patthar),
  plus schema facts `paudha needs paani`, `jaanvar needs khana`.
- **Multi-hop**: `entity → isa → category → needs → thing`, with no label at query
  time; the category is whatever `isa.recall` returns.
- **Metrics**: rank of the first target bit among all 512 output scores (hit@1, hit@3,
  MRR), plus mean overlap of the recalled SDR with the target.
- **Controls**: shuffled outputs (random values per fact) give chance-level scores.

## Bug found and fixed
`AssociativeMemory::recall` built its result as `SDR(out_, k_)`. The `SDR(n, k)`
constructor *randomly fills* k bits, so every recall returned the true top-k plus ~k
random bits (observed counts 43–48 instead of 24). Multi-hop fidelity was measured at
0.54 because the first hop fed a polluted code into the second memory. Fixed to
`SDR(out_, 0)` in `src/assoc.cpp`.

Consequence for earlier numbers: anything that used `recall`'s SDR as a *code* (not just
its argmax) was pessimistic. The argmax-based accuracies in `speech`, `wordspeech`,
`audit` were unaffected.

## Results

`./build/bin/curriculum` (hash entities):
```
A direct (fresh)   hit1=1.000 overlap=0.898
A multihop (fresh) hit1=1.000 overlap=1.000
A after B/C        hit1=1.000 overlap=0.898
B after C          hit1=1.000 overlap=0.894
C direct           hit1=1.000 overlap=0.694
A shuffled control hit1=0.000 overlap=0.069
```

`./build/bin/curriculum noreplay features`:
```
A after B   hit1=0.778 overlap=0.741   multihop overlap=0.521
A after C   hit1=0.889 overlap=0.847   multihop overlap=1.000
B after C   hit1=1.000 overlap=0.894
C direct    hit1=1.000 overlap=0.687
```

`./build/bin/curriculum replay features` (replay A 6x between B and C):
```
A after C   hit1=1.000 overlap=0.898
B after C   hit1=0.444 overlap=0.366
C direct    hit1=0.500 overlap=0.410
```

`./build/bin/curriculum interleave features` (A mixed 1:1 with each new fact):
```
A after C   hit1=1.000 overlap=0.898
B after C   hit1=0.778 overlap=0.759
C direct    hit1=0.667 overlap=0.590
```

Summary of the three replay policies (feature-bundle entities):

| policy | A | B | C |
|---|---|---|---|
| no replay | 0.889 | 1.000 | 1.000 |
| block replay A x6 | 1.000 | 0.444 | 0.500 |
| interleaved A:new = 1:1 | 1.000 | 0.778 | 0.667 |

Interleaving is the best compromise: it keeps the old set intact, loses less of the
middle set than block replay (0.778 vs 0.444), and only costs 0.333 on the newest set.

Reading:
1. **Multi-hop works exactly** when entity codes are orthogonal (overlap 1.000), and
   still degrades gracefully when codes overlap (0.52 after interference). The chaining
   is a real computation: `isa` output is fed as `needs` input, no labels involved.
2. **Shuffled control is at chance** (0.000 hit@1, 0.069 overlap vs 0.898). The memory
   stores the mapping, not the statistics of the label set.
3. **Feature-bundle entities forget**: A drops to 0.778 hit@1 after B. Hash entities do
   not forget at all, but they also cannot generalise — the two properties are the same
   knob.
4. **Facilitation**: without replay, A recovers to 0.889 after C, because `ped` shares
   `living/plant` bits with `gulab`. New related facts strengthen old rows.
5. **Replay is not free**: replaying A 6x fully protects A (1.000) but collapses the
   newer sets (B 0.444, C 0.500). Naive offline replay of old material is proactive
   interference, not a cure.
6. **Interleaved replay fixes most of it**: A 1.000, B 0.778, C 0.667. Mixing old and
   new facts in the same pass keeps the old rows alive without starving the new ones.

## Interpretation
The architecture has no separate "knowledge store": facts are weights in a Hebbian
matrix, and retention is decided by *representational overlap*. Orthogonal codes give
perfect retention and zero generalisation; overlapping codes give generalisation,
forgetting, and cross-set facilitation. Replay moves the balance but does not escape the
tradeoff. A biological system solves this with hippocampal separation + cortical
interleaved replay at low rate; that is the next experiment (replay interleaved with new
facts, and a separation pattern in the input).

## Not yet done
- Replay rate as a function of set size, and a separation layer before the matrix.
- A separation layer (pattern separator) before the matrix, so related entities do not
  collide.
- Fact revision (non-monotonic update: `gulab is-a patthar` should overwrite), and
  contradiction detection.
- Question answering over multi-hop chains longer than 2.