# 28 · Belief revision and reasoning depth

Two things `21_knowledge_induction.md` listed as not done: fact revision (non-monotonic
update) and question answering over chains longer than 2 hops. Both are now measured on
the existing `bm::AssociativeMemory`, with controls.

Everything below comes from `tools/curriculum.cpp` (experiments R and H, appended after the
existing experiments; no existing output was changed). Entity codes are SDRs from
`Encoder::symbol`; facts are Hebbian outer products in one 512x512 matrix per relation,
`lr = 0.05`, `k = 24`, decay `0.99999` per `learn()`.

## Method

**Experiment R (revision).** `gulab isa ped` is learned 5 times, then `gulab isa patthar`
is presented `m` more times, `m = 0, 1, 2, 5, 10, 20` (cumulative). After each step:

- `recall_old` = overlap of `isa.recall(gulab)` with `ped`, `recall_new` = same with `patthar`
  (both at the native `k = 24`).
- `recovered` = recall of `ped` after **one** re-presentation of `gulab isa ped` on a copy
  of the matrix (1 = overlap >= 0.5).
- `revision_wide` = both overlaps at `k = 48`, i.e. looking at the superposition instead of
  only the winner.
- `revision_control` = a second entity (`machli isa paani`, learned 5 times *before* the
  revision) recalled through the same matrix at each `m`.

**Experiment H (depth).** 24 chains of 5 entity codes `A -> B -> C -> D -> E`, one relation
(`hop`), each link learned once as `entity(A) -> entity(B)`. Query: start from `A`, feed
each hop's recall straight back in as the next input, and score the result against the true
node at that depth. No labels, no search, no beam. `top1` = first target bit at rank 0,
`overlap` = |recalled AND target| / |target|, `rank` = mean rank of the first target bit.

Both experiments run in both entity modes: `hash` (near-orthogonal random codes) and
`features` (bundles of shared feature bits).

## Experiment R: revision curve

`./build_curr/bin/curriculum` (hash entities), `k = 24`:

| m | recall_old | recall_new | recovered (1 rep) | old @ k=48 | new @ k=48 | control (machli) |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 0.000 | 1 | 1.000 | 0.042 | 1.000 |
| 1 | 1.000 | 0.000 | 1 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 0.000 | 1 | 1.000 | 1.000 | 1.000 |
| 5 | 0.125 | 0.875 | 1 | 1.000 | 1.000 | 1.000 |
| 10 | 0.000 | 1.000 | 0 | 1.000 | 1.000 | 1.000 |
| 20 | 0.000 | 1.000 | 0 | 1.000 | 1.000 | 1.000 |

Feature-bundle entities are identical except for two entries: at `m = 1` the wide recall of
the new fact is 0.167 instead of 1.000 (the new block is diluted by the old rows), and at
`m = 20` the control drops to **0.042**.

Reading:

1. **The curve is a step, not a decay.** The old belief stays at 1.000 until the new fact has
   been seen as often as the old one, then collapses within one step: at `m = 5` (5 vs 5, a
   tie in total evidence) the readout splits 0.125 / 0.875, and at `m = 10` the old fact is
   gone. There is no gradual weakening in between. This is what a linear sum of Hebbian
   outer products with a top-k readout does: the winner is decided by which block has the
   larger total weight, and every presentation adds the same quantum.
2. **Nothing is deleted.** At `k = 48` the old belief is at 1.000 for every `m`, including
   `m = 20`, and so is the new one: both answers sit in the matrix, fully intact, and which
   one you see is a function of the read-out width, not of storage. The memory does not
   revise beliefs; it **accumulates evidence and lets the readout compete**.
3. **Recovery is linear in re-presentations.** One re-presentation restores the old fact
   only while it was still competitive (`m <= 5`). After `m = 20` it takes **15**
   re-presentations to get back to overlap 0.9 (`curr,revision_recover`). Erasing the new
   belief is not possible at all: no presentation of `A isa B` ever subtracts from `A isa C`.

## Experiment R: controls

| control | result | verdict |
|---|---|---|
| recoverable? (`reps_to_overlap0.9`) | 15 after `m = 20` | decay, not deletion |
| locality (machli isa paani, same matrix) | 1.000 at every `m`, hash | local in weight space |
| locality, feature-bundle entities | 1.000 up to `m = 10`, **0.042 at `m = 20`** | **fails**: revision of `gulab` flips `machli` |
| permuted outputs (8 untrained codes, `m = 20`) | 0.036 overlap vs 0.047 chance | at chance |

The locality failure is a real interference effect, not noise. With feature bundles `gulab`
and `machli` share 8 input bits (the `living` feature). Each `gulab isa patthar`
presentation writes `0.05` into those 8 shared rows, so after 20 presentations the `patthar`
block scores `8 x 20 x 0.05 = 8.0` on `machli`'s input, against `24 x 5 x 0.05 = 6.0` for
its own `paani` block. `machli` then answers `patthar`. Revision is local in the weight
matrix but **not** local in the read-out once entity codes overlap: the more you revise one
belief, the more you damage its neighbours.

## Experiment H: hop depth

Chance for a 24-bit code in a 512-bit space: `top1 = overlap = 24/512 = 0.047`.

Isolated chains (one matrix per chain, no competing facts), 24 chains:

| depth | hash top1 | hash overlap | feature top1 | feature overlap |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | 0.991 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 1.000 | 1.000 | 1.000 | 0.990 |

Shared matrix (all 24 chains in one `hop` matrix, so each entity has ~12 different
successors - the realistic knowledge-base case):

| depth | hash top1 | hash overlap | hash rank | feature top1 | feature overlap |
|---|---|---|---|---|---|
| 1 | 0.667 | 0.443 | 0.88 | 0.792 | 0.493 |
| 2 | 0.458 | 0.326 | 3.21 | 0.750 | 0.481 |
| 3 | 0.625 | 0.332 | 4.50 | 0.875 | 0.493 |
| 4 | 0.500 | 0.316 | 5.21 | 0.750 | 0.491 |

Mean rank grows with depth (0.88 -> 5.21 hash) and overlap decays (0.443 -> 0.316), but the
degradation is caused by **conflicting facts in one matrix**, not by hop count: the first hop
is already down to 0.667. Depth per se is free.

### Controls for H

| condition | depth 3 | depth 4 | verdict |
|---|---|---|---|
| shuffled chain (random pairs, 24 chains x 4 facts, same count) | top1 0.000, overlap 0.038 | top1 0.083, overlap 0.045 | at chance |
| ablation: middle link `C -> D` removed, hash | top1 0.042, overlap 0.033 | top1 0.417, overlap 0.281 | hop after the gap is at chance |
| ablation: last link `D -> E` removed, hash | top1 1.000, overlap 1.000 | top1 0.042, overlap 0.036 | clean chance |
| ablation: middle link removed, feature | top1 0.333, overlap 0.217 | top1 0.833, overlap 0.366 | same |
| chance | 0.047 | 0.047 | |

Two honest caveats on the ablation rows:

- With the **middle** link missing, depth 3 is at chance but depth 4 recovers to 0.417-0.833.
  That is not reasoning. `recall` of a node with no outgoing link returns the same 24
  zero-score bits for every chain; that fixed set contains about one bit of the true next
  node, and a single shared bit already gives the whole successor block a score of
  `24 x 0.05`. One accidental bit is enough to name a block, so a gap can be bridged by
  luck two hops later. The **last-link** ablation is the clean test and it is at chance.
- `top1` on 24 chains is coarse (one chain = 0.042), so `overlap` is the number to read; the
  shuffled control's 0.083 at depth 4 is 2 chains out of 24, inside binomial noise for p =
  0.047.

### Why the chain uses one code per node

The existing `eval_multihop` feeds an `isa` **output** (`val:X`) into a memory keyed by
**entities** (`ent:X`). Those are different codes: over 64 symbols they share 1.16 bits on
average, and **32.8 % of pairs share none at all**
(`curr,hops_namespace,mean_shared_bits=1.16,zero_overlap_frac=0.328`). When the shared count
is 0, every output score is 0 and `recall` returns the same arbitrary block for any input, so
the hop is at chance; the 1.000 multi-hop numbers in doc 21 are consistent with that, i.e.
they depend on the two codes happening to collide in at least one bit. Experiment H
sidesteps this by using the same code for a node in both roles, which is what a knowledge
graph needs.

## Verdict

- **Does the memory revise beliefs?** Only in appearance. It never removes an association:
  a contradiction is added on top, and the read-out switches to the heavier block once the
  new fact has been seen at least as often as the old one (5 vs 5 is the break-even, 10 vs 5
  is already a clean takeover). The old belief survives in full at `k = 48`, and 15
  re-presentations bring it back to 0.9. Revision here is evidence weighting with a hard
  read-out threshold, not belief update.
- **Does it reason beyond 2 hops?** Yes, in the only sense this memory supports: iterated
  associative recall reproduces a 4-hop chain exactly (overlap 1.000, rank 0) as long as each
  hop's code is exact and the matrix holds one fact per node. The limit is not depth but
  **representational conflict**: with 24 chains in one matrix the first hop already drops to
  0.667 top-1 / 0.443 overlap and decays to 0.500 / 0.316 at depth 4. Add overlapping entity
  codes and revision of one fact silently rewrites a neighbour's answer (0.042 for `machli`
  after 20 revisions of `gulab`).
- **What it cannot do:** subtract (no unlearning, no contradiction detection), superpose two
  answers in a fixed-width read-out (both are present, the read-out picks one), or keep
  neighbours intact when codes share bits. Everything above is a property of `SDR`-in /
  `SDR`-out Hebbian accumulation with a top-k read-out, not of a missing module.

## Reproduce

```bash
cd /workspace/brain
cmake -S . -B build_curr -DCMAKE_BUILD_TYPE=Release
cmake --build build_curr -j4 --target curriculum
./build_curr/bin/curriculum
./build_curr/bin/curriculum noreplay features
```

Runtime 0.1 s, exit 0, no warnings under `-Wall -Wextra -Wpedantic`.

## Output format

No existing line changed: the pre-existing summary lines and the `A shuffled control` line
are byte-identical to the previous tool (verified by diff against a build of the previous
source). New `curr,revision*` and `curr,hops*` lines are appended after them. Beyond the keys
asked for (`curr,revision,m=,recall_old=,recall_new=,recovered=` and
`curr,hops,depth=,top1=,overlap=`) the following are added: `revision_re1` (old recall after
one re-presentation), `revision_wide` (both answers at `k = 48`), `revision_control` /
`revision_control_summary` (locality), `revision_permuted` (permuted outputs),
`revision_recover` (`reps_to_overlap0.9`), `rank` on every hop line, and the `hops_shared`,
`hops_ablation`, `hops_ablation_tail`, `hops_control`, `hops_chance`, `hops_namespace` rows.