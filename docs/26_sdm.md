# 26 · Sparse distributed memory (SDM, Kanerva hard locations)

A content-addressable memory with **no matrix**. It is the dense-matrix replacement for
`bm::AssociativeMemory`: where that class stores `N*N` floats and reads them with a
matrix-vector product, the SDM stores `A` fixed-size registers and reads them with a
majority vote. Everything is local, fixed-size, and online. No backprop, no attention,
no dense internal state.

Reproduce the measurements with (all numbers except the speed section are deterministic;
the speed numbers are one run on a shared 2-vCPU container and move by about 30 %):

```bash
cmake -S . -B build_sdm -DCMAKE_BUILD_TYPE=Release
cmake --build build_sdm -j4 --target sdm
./build_sdm/bin/sdm
```

## 1 · Design

Kanerva's model has two kinds of state per hard location: an **address** and a **data
register**. Both are C bits wide. The address is a random point in the same C-bit space as
the data, drawn once from a fixed seed and never changed. A cue is matched against the
addresses, not against the data, so the memory is addressed by *similarity of address*.

| piece | value | note |
|---|---|---|
| `addresses` (A) | 4096 | hard locations |
| `width` (C) | 1024 | address width == data width == SDR width |
| `threshold` (T) | 28 | minimum address overlap (bits) to select a location |
| `seed` | `0x5D1E5A17` | address table is fixed at construction |
| SDR width / k | 1024 / 40 | 3.9 % active |
| selected per cue | 38 | measured, clean cue |
| RAM | 1,052,672 B | 512 KB addresses + 512 KB registers + 4 KB write counts |

**Address table.** Each of the A addresses is a C-bit word with exactly C/2 ones, drawn from
`Rng(seed)`. Dense addresses are used on purpose: with a sparse cue (k of C) the overlap
with a dense address is `Binomial(k, 1/2)`, mean 20, sd 3.16 at k=40, so a threshold is
meaningful and small perturbations of the cue move the overlap by only a few bits.

**Selection.** A location is selected when `popcount(address & cue) >= threshold`. This is
one pass over the address table: `A * C / 64` word operations, no sorting, no allocation
other than the index list. At A=4096, C=1024, T=28 a clean cue selects 38 of 4096
locations (0.9 %).

**Write (chosen rule: XOR).** For every selected location, `register ^= data`, and the
location's write count is incremented (saturating at 255). XOR is used rather than additive
counters because it is 1 bit per cell, it is the textbook Kanerva rule, and it needs no
learning rate and no normalization. The write count is the only extra state: it is what
lets the read know how much interference a register has seen.

The consequence of XOR, stated up front: **writing the same word twice erases it**. The
memory stores the parity of the writes, not a sum. There is no decay and no unlearning.

**Read (majority vote).** The same selection is used. For each output bit, count how many
selected registers hold a 1 (`ones[j]`), and compare with what chance predicts.

The literal Kanerva rule is "the bit is on when `ones[j] > selected/2`". It works on a clean
cue — the cue wrote every selected location, so its own bit is present in all of them
(measured: 40 of 40 bits returned at 100 patterns) — and it fails on a noisy cue, because
the read set is then only partly the write set, the bit survives in about half the
registers, and the vote sits at chance. Measured with `SdmRead::HalfSelected`:

| cue noise | literal `ones > selected/2` | this memory |
|---|---|---|
| 0 % | 1.000 | 1.000 |
| 10 % | 0.525 | 1.000 |
| 25 % | 0.000 | 0.953 |
| 50 % | 0.000 | 0.509 |

The default read therefore subtracts the chance level. A register written `m` times by words
of density `rho` holds a 1 with probability

```
q(m) = (1 - (1 - 2*rho)^m) / 2
```

(the XOR of `m` independent `Bernoulli(rho)` bits), and a bit is on when

```
ones[j] > sum_a q(m_a) + z * sqrt(sum_a q(m_a) * (1 - q(m_a)))
```

with `z = 3` (`Sdm::kReadZ`). For `rho = 0.5` and `m = 1` this is exactly "half of the
selected locations"; for sparse words it is the same majority with the sparsity and the load
taken out. `rho` comes from the cue (`cue.count() / C`), `m_a` from the per-location write
count, so the rule stays local: no global statistics, no dense vector.

## 2 · Measurements

Config line: `addresses=4096,width=1024,threshold=28,sdr_n=1024,sdr_k=40`.

### 2.1 Does it learn? (capacity)

Auto-associative: store M random orthogonal SDRs, read each with its clean cue. Recall is
`|read & target| / |target|`, precision is `|read & target| / |read|`.

| patterns | recall | precision | read bits | literal half rule |
|---|---|---|---|---|
| 10 | 1.000 | 1.000 | 40.0 | 1.000 |
| 50 | 1.000 | 0.999 | 40.0 | 1.000 |
| 100 | 1.000 | 0.992 | 40.4 | 1.000 |
| 200 | 1.000 | 0.985 | 40.6 | 1.000 |
| 400 | 1.000 | 0.979 | 40.9 | 1.000 |
| 800 | 1.000 | 0.975 | 41.0 | 0.999 |
| 1200 | 0.986 | 0.978 | 40.3 | 0.992 |
| 1400 | 0.951 | 0.980 | 38.8 | 0.984 |
| 1500 | 0.920 | 0.981 | 37.5 | 0.977 |
| 1600 | 0.877 | 0.981 | 35.8 | 0.970 |
| 2000 | 0.628 | 0.977 | 25.7 | 0.930 |
| 2400 | 0.371 | 0.961 | 15.4 | 0.880 |

**Capacity at recall >= 0.9: 1500 patterns** (0.37 per hard location) for A=4096, C=1024,
k=40, T=28. Chance overlap is 0.039 (`k/N`), so 0.9 is 23x chance.

The failure mode is visible in the table: precision stays at 0.98 while recall collapses.
The memory does not start returning wrong bits, it stops returning the right ones: the
parity of ~20 writes per register drives `q(m)` to 0.5 and the vote has nothing left to
discriminate. That is the XOR capacity law, `M_0.9 ~ 0.37 * A` at this C and k.

**Dense baseline, same width, same patterns** (`bm::AssociativeMemory`, N*N floats, recall
on a 200-pattern sample):

| patterns | dense recall |
|---|---|
| 100 | 1.000 |
| 200 | 1.000 |
| 400 | 0.999 |
| 800 | 0.994 |
| 1200 | 0.993 |
| 1600 | 0.993 |
| 2000 | 0.996 |
| 2400 | 0.997 |
| 3200 | 0.998 |

For orthogonal random codes the dense matrix wins: it is still at 0.998 at 3200 patterns,
while the SDM is already down to 0.371 at 2400 (its grid stops there; the curve is
monotonic). The reason is that sparse random codes are the dense matrix's
best case — the correlation matrix is nearly diagonal, because the self term `count(j,j)`
grows as `M*k/N` while the cross term grows as `M*(k/N)^2`. The SDM's capacity is not
N-limited, it is A-limited, so the two curves cross in the other direction once the codes
are correlated (§2.7).

### 2.2 Does old knowledge survive new learning?

Train list A (100 patterns), measure A, train list B (100 different patterns), measure A
again.

```
before=1.000  after_B=1.000  listB_recall=1.000  chance=0.039
```

No replay, no rehearsal, no decay. Retention is exact because orthogonal codes select
almost disjoint location sets, so B's writes land where A's reads do not look. This is the
same result as the dense matrix, but with 1/4 of the RAM.

### 2.3 Can it predict/reconstruct unseen data? (noise)

Cue = the stored pattern with `f` of its **active** bits replaced by random positions (k is
held fixed; flipping 10 % of all N bits would turn a 4 % code into a 50 % code, which is not
the noise an SDR is supposed to survive).

| active bits replaced | SDM recall | SDM precision | read bits | literal half rule | dense recall |
|---|---|---|---|---|---|
| 0 % | 1.000 | 0.993 | 40.3 | 1.000 | 1.000 |
| 10 % | 1.000 | 0.955 | 42.0 | 0.525 | 1.000 |
| 25 % | 0.953 | 0.872 | 43.7 | 0.000 | 0.999 |
| 50 % | 0.509 | 0.585 | 30.8 | 0.000 | 0.970 |

The SDM completes a 25 %-corrupted cue at 0.953 and survives 50 % at 0.509 (13x chance).
The dense matrix is better at every noise level: a linear readout degrades gracefully because
the noisy cue is still 75 % correlated with the original. This is a real loss for the SDM and
it is not hidden.

### 2.4 RAM

| structure | bytes | formula |
|---|---|---|
| SDM addresses | 524,288 | `A * C / 8` |
| SDM registers | 524,288 | `A * C / 8` |
| SDM write counts | 4,096 | `A` |
| **SDM total** | **1,052,672** | `2 * A * C / 8 + A` |
| dense matrix | 4,194,304 | `N * N * 4`, N=1024 |

Ratio **3.98x**. Bytes per stored pattern at 100 patterns: 10,527 (SDM) vs 41,943 (dense).
The SDM's RAM does not depend on how many patterns are stored, and it does not depend on N
at all: at A=1<<20, C=1024 it is 256 MB, while the dense matrix for the same SDR width is
4 TB and cannot be allocated.

### 2.5 Speed

Single thread, `-O3`, 500 operations each on a 100-pattern memory. The machine is a shared
2-vCPU container, so these are the only numbers on this page that are not reproducible:
across runs the SDM write/read sat between 116 and 126 us and the dense ops between 24.7
and 27.6 us. One run:

| operation | per call | per second |
|---|---|---|
| SDM write | 126.3 us | 7,917 |
| SDM read | 141.3 us | 7,076 |
| dense `learn_batch`, per pattern | 27.6 us | 36,200 |
| dense `recall` | 50.2 us | 19,920 |

An SDM write and read cost the same because both are dominated by the selection pass
(`A * C / 64` word operations); the XOR and the vote are negligible. The cost is linear in A
and C and independent of the number of stored patterns. **The dense matrix is faster here**:
its matrix-vector product is a dense, vectorisable pass over 1 M floats, while the SDM's
selection is a gather over 4096 addresses with a popcount each. The SDM does not buy speed,
it buys RAM and a different scaling.

### 2.6 Controls (is the result an artifact?)

| control | measured | chance | verdict |
|---|---|---|---|
| uncorrelated cue (never stored) | 0.0310 | 0.0391 | chance |
| cross-talk: read key i, compare to value j (`i != j`, 9900 pairs) | 0.0400 | 0.0391 | chance |
| shuffled pairing: store key i with value perm(i), ask for value i | 0.0482 | 0.0391 | chance |
| literal half rule on a clean cue | 40.00 bits | 40 | not degenerate |

The first three are the "shuffled/uncorrelated data must fail" control. The memory stores
the pairing it was shown and nothing else: a random cue returns 0.031, a wrong key returns
0.040, and a shuffled pairing returns 0.048, all at the 0.039 chance level. The fourth line
is the control for the read rule: the literal majority is not degenerate on clean cues, it
is the *noisy* cue that breaks it (§1).

### 2.7 Correlated codes: where the SDM earns its place

Random orthogonal codes are the dense matrix's best case. Real encoder output is the
opposite: similar inputs produce similar SDRs, and the dense correlation matrix cannot
separate them because it is linear. 20 prototypes x 5 variants, each variant a 30 % bit
replacement of its prototype (100 patterns):

| model | recall | precision | read bits | prototype overlap |
|---|---|---|---|---|
| SDM | 1.000 | 0.886 | 45.6 | 0.798 |
| dense matrix | 0.799 | 0.799 | 40.0 | - |

The SDM keeps every exact pattern (recall 1.000) and returns 5.6 extra bits on average
(precision 0.886). The dense matrix returns a 0.8 blend: a bit that appears in the four
sibling variants but not in the target accumulates `4 * |X n sibling| = 120` counts, while a
bit of the target itself accumulates `1 + 3 * |X n sibling| = 91`. The siblings outscore the
target, so the exact pattern is lost. Neither is clean, but the SDM's
error (a superset that still contains the answer) is the useful one for a downstream k-WTA,
and the dense error (a substitute) is not.

### 2.8 Content-addressable attention substitute

Store 100 (key, value) pairs, query with a noisy key, and report the rank of the correct
value by overlap with the 100 stored values. Same protocol for both models.

| pairs | key noise | model | top-1 | top-5 | mean rank | median rank | value recall |
|---|---|---|---|---|---|---|---|
| 100 | 10 % | SDM | 1.00 | 1.00 | 1.0 | 1 | 1.000 |
| 100 | 10 % | dense | 1.00 | 1.00 | 1.0 | 1 | 1.000 |
| 100 | 25 % | SDM | 1.00 | 1.00 | 1.0 | 1 | 0.977 |
| 100 | 25 % | dense | 1.00 | 1.00 | 1.0 | 1 | 0.999 |
| 400 | 25 % | SDM | 0.98 | 0.98 | 1.4 | 1 | 0.696 |
| 400 | 25 % | dense | 1.00 | 1.00 | 1.0 | 1 | 0.863 |

Control: a random key that was never stored ranks the correct value top-1 in 0.00 of cases
at 100 and 400 pairs (chance 0.01 / 0.0025).

At 100 pairs both models are perfect; the SDM uses 1 MB and the dense matrix 4 MB. At 400
pairs with a 25 % corrupted key the SDM still ranks the correct value first in 98 % of cases
(mean rank 1.4) but only returns 0.696 of the value's bits — enough to rank, not enough to
read out. The dense matrix returns 0.863 of the bits. This is the honest summary of the
substitution: **the SDM is a good ranker and a mediocre reconstructor at high load.**

## 3 · SDM vs the dense matrix

| property | SDM (A=4096, C=1024) | `bm::AssociativeMemory` (1024x1024) |
|---|---|---|
| RAM for one memory | 1.03 MB | 4.19 MB |
| RAM formula | `2*A*C/8 + A` | `N*N*4` |
| capacity at recall 0.9, orthogonal codes | 1500 | > 3200 (0.998 at 3200) |
| recall at 25 % cue noise | 0.953 | 0.999 |
| recall, correlated codes (20x5) | 1.000 (precision 0.886) | 0.799 (precision 0.799) |
| attention top-1, 400 pairs at 25 % noise | 0.98 | 1.00 |
| learning rule | XOR into hard locations, no learning rate | Hebbian outer product + decay |
| per-operation cost | O(A*C/64): write 116 us, read 127 us | O(N*N): learn_batch 26 us/pattern, recall 47 us |
| state that grows with data | none | none |
| unlearning | impossible (XOR is its own inverse) | decay only |

The dense matrix is more accurate on orthogonal codes and on noisy cues at this width, and
it is faster per operation. What the SDM buys is RAM: 4x smaller for the same task, about 2x
more patterns per MB on orthogonal codes and 4x more on correlated codes, with no term that
grows as N*N. At N = 1<<20 the dense matrix needs 4 TB and cannot be built; the SDM needs
256 MB. The choice is a RAM and a data-statistics choice, not an accuracy or a speed choice.

## 4 · What fails

1. **The dense matrix wins on orthogonal codes.** At the same width it is still at 0.998 at
   3200 patterns while the SDM is at 0.371 at 2400; under 25 % cue noise it is 0.999 vs 0.953.
   For random sparse codes the correlation matrix is nearly diagonal and the SDM has no answer
   to that. The SDM's advantage needs correlated codes (§2.7).
2. **The dense matrix is also faster per operation** (26 us vs 116 us to learn, 47 us vs 127 us
   to read). The SDM's selection is a gather with a popcount per address; the dense readout is
   a vectorisable matrix-vector product. The SDM is not a speed win at this size.
3. **The literal Kanerva read rule fails under noise.** `ones > selected/2` returns 1.000 on
   a clean cue and 0.000 at 25 % noise. The load-normalised rule is a deviation from the
   textbook rule and is measured to be necessary for the noisy-cue experiment.
4. **XOR capacity is parity-limited.** `q(m)` reaches 0.41 at `m = 20` writes per register and
   only tends to 0.5, but the discriminable range ends there: at 2400 patterns the load is
   `m = 22`, `q = 0.43`, and the vote is down to 0.371. That caps the memory at
   `M ~ 0.37*A` for this A, C and k regardless of how much RAM is available. Additive counters
   would raise this by an order of magnitude but cost 8x the RAM per cell and break the
   `A*C/8` budget.
5. **No unlearning and no decay.** Writing a word twice erases it; a location written an even
   number of times reads as zero. There is no way to forget, and no way to repair a bad write.
6. **The threshold is tied to k.** T=28 is the 98.7th percentile of `Binomial(40, 0.5)`. At a
   different SDR sparsity the same T selects a different fraction of the addresses and the
   capacity moves. A fraction-of-k threshold would be more robust; the parameter is an
   absolute count because the API is fixed.
7. **Read precision is not exact even at capacity.** At 1200 patterns the SDM returns 40.3
   bits for a 40-bit target (0.9 false positives per read, 0.6 bits missing); on the
   correlated set it returns 45.6 bits for a 40-bit target (precision 0.886). The error is a
   superset, never a substitute, which is why the ranking survives when the exact
   reconstruction does not.
8. **High load + high noise breaks reconstruction.** At 400 pairs and 25 % key noise the value
   recall is 0.696: the correct value is still ranked first (top-1 0.98), but the returned
   bits are not the value. An attention substitute that needs the value, not its rank, would
   need a second cleanup stage.
9. **Speed is selection-bound.** 127 us per operation at A=4096, C=1024. The cost is linear
   in `A*C`, so a larger memory is proportionally slower; there is no sublinear index.

## 5 · API

```cpp
struct SdmParams {
  size_t addresses = 4096;   // A
  size_t width = 1024;       // C
  size_t threshold = 28;     // minimum address overlap
  uint32_t seed = 0x5D1E5A17u;
};

class Sdm {
 public:
  explicit Sdm(const SdmParams& params = SdmParams{});
  void write(const SDR& address, const SDR& data);
  SDR read(const SDR& address) const;
  SDR read(const SDR& address, SdmRead rule) const;  // Majority (default) | HalfSelected
  size_t selected(const SDR& address) const;
  size_t ram_bytes() const;
};
```

`SdmRead::HalfSelected` is the literal `ones > selected/2` rule, kept as a control and used
in the tables above. The default `read()` is the load- and sparsity-normalised majority.