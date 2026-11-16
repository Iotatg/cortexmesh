#pragma once

#include <cstdint>
#include <vector>

#include "bm/sdr.hpp"

namespace bm {

struct TemporalMemoryParams {
  size_t column_count = 2048;
  size_t cells_per_column = 32;
  size_t initial_segments = 0;
  size_t max_segments_per_cell = 16;
  uint32_t activation_threshold = 2;
  uint32_t min_threshold = 2;
  uint32_t new_segment_min = 3;
  double learning_radius = 0.5;
  uint8_t permanence_inc = 2;
  uint8_t permanence_dec = 2;
  uint8_t permanence_threshold = 12;
  uint64_t seed = 11;
  bool use_predictive_in_prev = true;
};

struct Segment {
  std::vector<uint32_t> cells;
  std::vector<uint8_t> perms;
  uint8_t age = 0;
};

struct TMStep {
  SDR active;
  SDR predictive;
  size_t bursting_columns = 0;
  size_t predicted_active = 0;
};

class TemporalMemory {
 public:
  explicit TemporalMemory(const TemporalMemoryParams& p);

  TMStep step(const SDR& active_columns);

  void set_learning_gain(float g) { learn_gain_ = std::max(0.0f, g); }
  float learning_gain() const { return learn_gain_; }
  double anomaly_score() const { return last_anomaly_; }
  double prediction_accuracy() const { return last_accuracy_; }
  size_t segment_count() const;
  void reset();

  size_t cell_count() const { return cells_; }
  size_t column_of(uint32_t cell) const { return cell / p_.cells_per_column; }

  const std::vector<std::vector<Segment>>& segments() const { return segs_; }
  const SDR& predictive() const { return last_predictive_; }
  const SDR& previous_active() const { return prev_active_; }

 private:
  size_t active_synapses(const Segment& s, const SDR& prev) const;
  void adapt_segment(Segment& s, const SDR& prev, float polarity);
  void grow_segment(uint32_t cell, const std::vector<uint32_t>& prev_idx);

  TemporalMemoryParams p_;
  size_t cells_;
  std::vector<std::vector<Segment>> segs_;
  SDR prev_active_;
  SDR prev_predictive_;
  SDR last_predictive_;
  double last_anomaly_ = 0.0;
  double last_accuracy_ = 0.0;
  Rng rng_;
  float learn_gain_ = 1.0f;
};

}  // namespace bm