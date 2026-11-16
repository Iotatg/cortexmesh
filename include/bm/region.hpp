#pragma once

#include <cstdint>

#include "bm/sdr.hpp"
#include "bm/spatial_pooler.hpp"
#include "bm/temporal_memory.hpp"

namespace bm {

struct RegionParams {
  size_t input_size = 1024;
  size_t column_count = 1024;
  size_t cells_per_column = 16;
  size_t synapses_per_column = 64;
  size_t active_columns = 20;
  uint64_t seed = 3;
  double feedback_gain = 0.0;
  bool use_predictive_in_prev = true;
  uint32_t activation_threshold = 2;
  uint32_t new_segment_min = 3;
};

struct RegionStep {
  SDR input;
  SDR columns;
  SDR active;
  SDR predictive;
  size_t bursting_columns = 0;
  size_t predicted_active = 0;
  double anomaly = 0.0;
};

class Region {
 public:
  explicit Region(const RegionParams& p);

  RegionStep step(const SDR& input);
  RegionStep step(const SDR& input, const SDR& feedback);

  double prediction_accuracy() const { return tm_.prediction_accuracy(); }
  double anomaly() const { return tm_.anomaly_score(); }
  size_t segment_count() const { return tm_.segment_count(); }
  void set_focus(float gain) {
    tm_.set_learning_gain(gain);
    sp_.set_tie_jitter(0.02f * gain);
  }

  SDR predicted_columns() const;
  const SDR& stable_output() const { return stable_; }
  SDR& stable_output() { return stable_; }

 private:
  static SpatialPoolerParams MakePoolerParams(const RegionParams& p);
  static TemporalMemoryParams MakeTmParams(const RegionParams& p);

  RegionParams p_;
  SpatialPooler sp_;
  TemporalMemory tm_;
  SDR stable_;
  double stable_rate_ = 0.0;
};

}  // namespace bm