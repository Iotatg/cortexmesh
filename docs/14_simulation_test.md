# 14 · Simulation test: does the AI notice it is inside a simulation?

A deterministic grid world + a `CortexMesh` region that learns the world from observation
alone, then a set of rule violations ("glitches"). If the AI has a world model, a glitch must
raise its prediction failure (burst) rate. No labels are used.

## World

`tools/sim.cpp`:

- 24×24 grid, one agent, 3 food items, deterministic movement (reflect on walls).
- Two observation streams:
  - **patch** (appearance): 9×9 local patch (walls + food) + distance to nearest wall.
  - **pose** (where): a hash-based state assembly `state:<id>` for `(y, x, dir)`, 20/4096 active.
    Not a language token: the language path uses feature-grounded word codes (doc 15).
- Two independent regions: `patch_ai` (input 2048) and `state_ai` (input 4096).
- Trained on 300-step clean runs, 20 epochs, per condition (fresh AIs each time).

## Result (final)

```
condition,patch_burst,patch_run,state_burst,state_run,pred_acc,verdict
clean,0.000,0.00,0.000,0.00,1.000,consistent
teleport,0.340,2.00,0.930,18.60,0.812,ANOMALY
pass_wall,0.000,0.00,0.010,1.00,1.000,consistent
speed2,0.000,0.00,0.390,3.00,1.000,ANOMALY
speed3,0.000,0.00,0.310,2.38,1.000,ANOMALY
spin,0.000,0.00,0.050,2.50,1.000,consistent
food_dup,0.000,0.00,0.000,0.00,1.000,consistent
noise1,0.002,1.00,0.000,0.00,0.999,consistent
noise3,0.003,1.00,0.000,0.00,0.998,consistent
summary,clean=0.000,false_positive=0.000,detection_floor=0.000,margin=0.000,detected=3/6
```

Note: `state_burst` is now the **fraction** of active columns bursting, not a binary flag,
so absolute values shifted down (teleport 0.960 -> 0.930). Verdicts are unchanged.

## The main finding: the SP needs near-orthogonal inputs

Before the single-token state code, the pose was encoded as factored sub-codes. The SP then
produced identical columns for different positions (overlap 20/20) and a 2-cell jump produced zero
bursts — see D13. After switching to a single sparse code per state, clean burst fell to 0.017 and
teleport rose to 0.96 and clean burst fell to 0.000.

Trying to replace the state id with a **grid-cell place code** failed for a deeper reason (D16):
neighbouring grid codes overlap ~50 %, so SP columns overlap and the TM cannot assign a unique
successor (clean burst 0.19–0.77, teleport 0.15–0.23). Random/orthogonal codes give unique
columns. The SP/TM stack needs orthogonal inputs or a much larger pooler.

## Original finding: spatial pooler collapses to the invariant part

First attempt encoded the pose as **factored sub-codes** (x code + y code + dir code + food
code). The SP then produced **identical column codes for different positions**:

```
code overlap 4-6 = 20/40   (x differs, y/dir/food same)
column overlap 4-6 = 20/20 (identical SP output)
jump-by-2 burst = 0/11     (the AI cannot see a 2-cell jump)
```

Cause: the constant components (y, dir, food) give many columns a stable overlap, so the
top-20 winners are the same columns for every position. The SP learns a **pose-invariant**
code and the position information is discarded.

Fix: encode the **whole state as one token** (no shared sub-codes). Then different positions
have disjoint codes, and the pose stream works:

| | factored sub-codes | single token |
|---|---|---|
| column overlap, different positions | 20/20 | 0/20 |
| clean burst | 0.00 | 0.003 |
| jump-by-2 burst | 0.00 | 0.63 |

This is a general rule for `CortexMesh`: **do not concatenate sub-codes that share a constant
component**; the SP will collapse onto the shared part. Use a conjunction/token for the full
state, or separate regions per stream.

## What the two streams buy

| | pose stream | patch stream |
|---|---|---|
| clean | 0.003 | 0.000 |
| sensor noise (1–3 bit flips) | 0.003 | 0.000–0.007 |
| teleport | 0.950 | 0.580 |
| speed 2 / 3 | 0.630 / 0.610 | 0.000 |
| spin | 0.110 | 0.000 |
| pass_wall | 0.010 | 0.000 |
| food_dup | 0.000 | 0.000 |

- **Pose stream = "where"**: detects kinematic rule violations (teleport, speed) with almost
  zero false positives on sensor noise. This is the useful simulation detector.
- **Patch stream = "what"**: nearly constant in open space, so it misses food/wall events.
  It needs a richer encoding (multi-scale, food proximity) before it is useful.

## Honest limits

1. **Rare events are diluted by rate.** `pass_wall` wraps once every ~24 steps; the burst is
   real but the rate over 100 steps is 0.01. Use an event-window or persistence (`run_max`)
   metric: teleport `run_max = 15.8` vs noise `1.0` is a clean separation.
2. **Spin is weak (0.11).** The heading token changes every step, but the pose stream
   re-learns quickly; a longer window or a heading-specific stream would help.
3. **Food is invisible to the pose stream.** Food count/position is not in the pose token;
   the patch stream is too weak to see it. A dedicated "object" stream is needed.
4. **No action input.** The model predicts the next state from the current state only. A
   forward model `(state, action) -> next state` would separate "the world changed" from
   "I moved differently".

## How to run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/bin/sim            # 20 epochs, 300 steps
./build/bin/sim 30 400 1 2 16   # epochs steps use_pred act_thresh cells
```

`sim_smoke` (2 epochs, 40 steps) is part of `ctest`.
