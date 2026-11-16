# 10 · Public API (`bm`)

All types live in namespace `bm`. Build target: `bm` (static lib), `brain_cli`, `test_core`.

## `sdr.hpp`

```cpp
class SDR {
  SDR(size_t n, size_t k = 0);
  static SDR random(size_t n, size_t k, uint64_t seed);
  static SDR from_indices(size_t n, const std::vector<uint32_t>& idx);
  size_t size(), count();
  void set(uint32_t), reset(uint32_t), clear(), fill_k(uint32_t, Rng&);
  bool test(uint32_t) const;
  SDR operator& | ^ ~ ;
  size_t overlap(const SDR&) const;
  double jaccard(const SDR&) const;    // intersection / union
  double similarity(const SDR&) const; // intersection / self count
  std::vector<uint32_t> indices() const;
  std::string to_string() const;
  uint64_t hash() const;
};
```

Invariants: `count() <= size()`; `overlap(a,a) == a.count()`; `indices()` sorted ascending.

## `encoder.hpp`

```cpp
class Encoder {                    // compositional, deterministic, cache-backed
  SDR symbol(const std::string&);
  SDR token(uint32_t id);
  SDR context(const std::vector<std::string>& history, size_t order);
  SDR conjunction(const std::vector<SDR>& parts);
  SDR grid(double x, double y, size_t modules, size_t bits_per_module);
};

class ScalarEncoder {              // overlapping receptive fields
  SDR encode(double v) const;      // monotone: |v1-v2| small => overlap high
};
```

## `spatial_pooler.hpp`

```cpp
struct SpatialPoolerParams {
  size_t input_size, column_count, synapses_per_column;
  double connected_permille;
  size_t active_columns;
  uint8_t permanence_threshold, permanence_init_min, permanence_init_max;
  uint8_t permanence_inc, permanence_dec;
  int growth_prob;                    // percent chance per win to add one synapse
  size_t growth_min_size;             // 0 => synapses_per_column / 3
  double boost_max, boost_inc, boost_decay, tie_jitter;
  uint64_t seed;
};

class SpatialPooler {
  SDR activate(const SDR& input);     // learns on the winners
  SDR activate(const SDR& input, const SDR& feedback);  // apical bias (feedback_gain)
  const std::vector<float>& boosts() const;
  size_t dead_columns() const;
  size_t probe_dead(const std::vector<SDR>& probes) const;
  size_t mean_synapses() const;
  size_t prune(float min_perm);       // drop weak synapses, keep a connectivity floor
  void reset_homeostasis();
};
```

Guarantees: output is `active_columns` sparse bits (or fewer if no column has overlap);
each column owns exactly `synapses_per_column * connected_permille / 1000` synapses;
total synaptic weight per column is conserved (homeostasis).

## `temporal_memory.hpp`

```cpp
struct TemporalMemoryParams {
  size_t column_count, cells_per_column, initial_segments, max_segments_per_cell;
  uint32_t activation_threshold, min_threshold, new_segment_min;
  uint8_t permanence_inc, permanence_dec, permanence_threshold;
  uint64_t seed;
};

struct Segment { std::vector<uint32_t> cells; std::vector<uint8_t> perms; uint8_t age; };

struct TMStep { SDR active, predictive; size_t bursting_columns, predicted_active; };

class TemporalMemory {
  TMStep step(const SDR& active_columns);   // one tick: predict, activate, learn
  const SDR& predictive() const;            // last step's predicted cells
  double anomaly_score() const;             // 1.0 if the tick bursted
  double prediction_accuracy() const;       // predicted / active this tick
  size_t segment_count() const;
  void reset();
};
```

Contract: `step` is online and order-dependent; a column that bursts means "this transition
was not predicted". After `k` exposures to a fixed sequence, `bursting_columns == 0` for
all transitions (see `11_benchmarks.md`).

## `region.hpp`

```cpp
struct RegionParams { size_t input_size, column_count, cells_per_column,
                      synapses_per_column, active_columns; uint64_t seed;
                      double feedback_gain; };

struct RegionStep { SDR input, columns, active, predictive;
                    size_t bursting_columns, predicted_active; double anomaly; };

class Region {
  RegionStep step(const SDR& input);
  RegionStep step(const SDR& input, const SDR& feedback);  // feedback = column code
  SDR predicted_columns() const;                           // TM prediction as columns
  double prediction_accuracy() const;
  double anomaly() const;
  size_t segment_count() const;
  const SDR& stable_output() const;   // temporally pooled code
};
```

A region = spatial pooler + temporal memory + temporal pooling. This is the unit that
hierarchies are built from.

## `valuation.hpp`

```cpp
class Valuation {
  Valuation(size_t n = 4096, float alpha = 0.05f, float trace_decay = 0.85f);
  float predict(const SDR& state) const;   // mean weight over active bits minus baseline
  void present(const SDR& state);
  void reward(float r);                    // RPE update over the eligibility trace
  void tick();                             // decay weights and baseline
  float rpe() const;
  float weight(size_t i) const;            // learned weight of bit i
};
```

Sparse linear readout with a moving baseline. See D3/D6 in `13_discoveries.md`: use short
traces and per-action heads.

Limitation (X2): a linear head cannot represent conjunctions (e.g. state parity × body).
For conjunctive rules, add the body bits as extra inputs (interoception) or route the
conjunction through a region.

## `hippocampus.hpp`

```cpp
class Hippocampus {
  SDR dentate(const SDR& cortical) const;         // random expansion, top-k
  void store(const SDR& cortical);
  SDR complete(const SDR& cue, float min_overlap = 0.5f) const;
  float mismatch(const SDR& a, const SDR& b) const;
};
```

 Dentate = fixed random projection with `fan_in` inputs per dentate cell, top-k winners.
`complete` returns the dentate code of the stored cortical pattern closest to the cue.

## `dual_controller.hpp`

```cpp
class DualController {
  void learn_model_free(const SDR& state, const SDR& action, float reward);
  void learn_model_based(const SDR& state, const SDR& action, const SDR& outcome,
                         float outcome_reward);
  Decision decide(const SDR& state, const std::vector<SDR>& actions, float uncertainty);
  float model_free_value(const SDR& state, const SDR& action) const;
  float model_based_value(const SDR& state, const SDR& action) const;
};
```

Per-action value heads (`action_values_[action.hash()]`), not a shared conjunction readout
(see D6). Model-based branch uses stored outcomes when uncertainty is high.

## `modulator.hpp`

```cpp
struct Modulators { float rpe, novelty, ach, na; float gain() const; };

class ModulatorSystem {
  void observe_state(uint64_t state);
  void observe_reward(float reward);      // dopamine RPE proxy
  void observe_anomaly(float anomaly);    // noradrenaline / novelty proxy
  void set_interoception(float arousal, float valence);
  void tick();
  const Modulators& modulators() const;
  float value(), uncertainty(), predicted_reward() const;
};
```

`rpe = r + gamma V(s') - V(s)`; `novelty` is an EMA of anomaly; `ach` rises with
uncertainty (focus/precision); `na` spikes when novelty exceeds `na_theta`.

## `interoception.hpp`

```cpp
struct BodyState { float heart_rate, hrv, breathing, effort; };

class Interoception {
  SDR encode(const BodyState&) const;    // 4 scalar channels -> SDR
  float arousal(const BodyState&) const; // HR + breathing
  float valence(const BodyState&) const; // HRV
};
```

## `salience.hpp`

```cpp
class Salience {
  void update(float anomaly);
  float gain() const;                    // 1 .. max_gain
  float learning_rate(float base_lr) const;
  float anomaly(), uncertainty() const;
};
```

Gain rises with anomaly and uncertainty and decays back to baseline. This is the
computational "salience network": it decides how strongly the current event is processed.

## Focus wiring

`Region::set_focus(float g)` sets the temporal-memory learning gain and the spatial
pooler's tie jitter. Moderate focus sharpens learning; extreme focus adds noise
(inverted-U, matching the LC-NE adaptive-gain account).

## `tools/brain_cli.cpp`

```
brain_cli encode <text>
brain_cli seq
```

`seq` trains a 32-item random sequence through a 4096-column region and prints accuracy and
the anomaly score for an unseen item.

## Extension points (not yet implemented)

| Header | Will provide |
|---|---|
| `bm/modulator.hpp` | `Modulators { rpe, novelty, ach }` broadcast scalars |
| `bm/plasticity.hpp` | eligibility traces, three-factor update, tagging |
| `bm/replay.hpp` | compressed replay buffer, consolidation pass |
| `bm/hippocampus.hpp` | dentate expansion, CA3 attractor, CA1 mismatch |
| `bm/basal_ganglia.hpp` | action WTA, RPE-driven values |
| `bm/cerebellum.hpp` | granule expansion, climbing-fiber error, forward model |