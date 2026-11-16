# Context-free retrieval: old information without a fixed window

Question: can a fixed-size associative memory bring information learned long ago into a
*new* context, without a fixed context window?

Short answer: **yes for single items, no for deep chains.** At a fixed load the memory's
recall is flat from age 0 to age 1024 (0.89-0.91), where a 64- or 256-item window drops to
0.000. But iterated retrieval compounds error, so a 5-link chain through the same memory
ends at 0.00, and reusing an old item in a new context needs a separate read-out per
context (0.69 vs 0.05 for a wrong key).

Tool: `tools/contextfree.cpp` (`./bin/contextfree`). Runtime 1.3 s, peak RSS ~9 MB.

## 1. Capacity sets the ceiling, not recency

Unique pairs learned in one batch, 1024x1024 float matrix, k=24.

| items | recall |
|---|---|
| 100 | 1.000 |
| 500 | 0.9997 |
| 1000 | 0.970 |
| 2000 | 0.828 |
| 5000 | 0.515 |
| 10000 | 0.295 |
| 20000 | 0.183 |

## 2. Age curve: no window cliff

1500 unique items streamed one at a time (one decay per item), then each item queried at
its age. Memory recall vs the same query answered by a fixed window of the last W items.

| age | memory recall | window 64 | window 256 |
|---|---|---|---|
| 0 | 0.912 | 1.000 | 1.000 |
| 8 | 0.914 | 1.000 | 1.000 |
| 32 | 0.905 | 1.000 | 1.000 |
| 64 | 0.905 | 0.000 | 1.000 |
| 128 | 0.897 | 0.000 | 1.000 |
| 256 | 0.912 | 0.000 | 0.000 |
| 512 | 0.891 | 0.000 | 0.000 |
| 1024 | 0.894 | 0.000 | 0.000 |

The window is a step function; the memory is flat. At this load the answer to "how far back
can it reach" is "all of it, at 0.90", and the cost of distance is paid in load, not in age.

## 3. Deep chains break

100 chains of 5 links plus 500 distractor facts in one memory (612 pairs).

| depth | all hops correct | + cleanup memory |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 0.750 | 0.990 |
| 3 | 0.090 | 0.990 |
| 4 | 0.020 | 0.990 |
| 5 | 0.000 | 0.990 |

Cleanup memory (512 KB for 2600 codes) snaps each retrieved code back to the closest known
code before it is used as the next cue. It changes 495 of 500 retrieved codes and every
change is correct. Denoise accuracy is 1.000 up to 12 flipped bits out of 16, and 0.000 at
14 (the code is then closer to a different item). Cost: 29.7 us per clean at 4096 items.

Histogram of how far chains get: 2 links 66, 3 links 7, 4 links 2, 5 links 0.

Per-hop recall is ~0.90, but the retrieved code is *approximate*: a 10% bit error in the
retrieved item is a 10% error in the next cue, and the errors compound. Independent errors
would predict 0.9^5 = 0.59, so the real decay is super-linear - the read-out is the
bottleneck, not the chaining.

## 4. Old item in a new context

Region A: `a -> b` (200 pairs). Region B: `(b + role) -> v` (4 contexts x 50 items).
Query: retrieve `b` from `a`, then use the retrieved `b` with a *new* role tag to get `v`.

| condition | end-to-end |
|---|---|
| retrieved b -> new role -> v | 0.692 |
| true b -> new role -> v (upper bound) | 0.692 |
| wrong b -> new role -> v | 0.054 |
| b retrieval accuracy | 1.000 |
| control: all roles in one shared read-out | 0.650 |

The old item is fully recovered (1.000) and works in a new context (0.69 vs 0.05 for a
wrong item). Two things limit it: read-out capacity (50 items per context) and the fact
that a *shared* role code floods the matrix - an 8-bit role tag reused across 200 items
adds a background of ~1.9 per output bit against a signal of ~1.6, which is why the first
single-matrix version scored 0.26.

## 5. Controls

| control | recall |
|---|---|
| shuffled values | 0.032 |
| untrained matrix | 0.032 |
| wrong key in new context | 0.054 |

## Verdict

1. **Yes, without a window:** a fixed 1 MB matrix answers items 1024 steps old at 0.89, and
   the state does not grow with context length (RAM is constant at any age).
2. **The limit is load, not time:** accuracy falls with the number of stored items
   (0.97 at 1k, 0.51 at 5k), and stays flat across age at fixed load.
3. **Deep chains were broken, and cleanup fixes them:** 5-hop iterated retrieval is 0.00 raw
   because each retrieved code is approximate and the error compounds. Snapping each retrieved
   code to the nearest known code (cleanup memory, 512 KB) takes 5-hop to 0.990.
4. **New contexts need their own read-out:** a shared role code inside one matrix is
   swamped by its own background (0.26); one read-out per context gives 0.69.

Next step: wire cleanup into every re-cue path (hippocampal loop, language model, sim pose
codes) and measure whether the gain holds when the code set is large (100k items).