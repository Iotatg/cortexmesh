#pragma once

#include <cstdint>
#include <vector>

#include "bm/sdr.hpp"

namespace bm {

struct Syn {
  uint32_t idx = 0;
  uint8_t w = 0;
};

struct SpatialPoolerParams {
  size_t input_size = 4096;
  size_t column_count = 2048;
  size_t synapses_per_column = 128;
  double connected_permille = 500;
  size_t active_columns = 40;
  uint8_t permanence_threshold = 12;
  uint8_t permanence_init_min = 12;
  uint8_t permanence_init_max = 30;
  uint8_t permanence_inc = 1;
  uint8_t permanence_dec = 1;
  int growth_prob = 50;
  size_t growth_min_size = 0;  // 0 => synapses_per_column / 3
  double feedback_gain = 0.0;
  double boost_max = 1.0;
  double boost_inc = 0.0;
  double boost_decay = 0.02;
  double tie_jitter = 0.02;
  uint64_t seed = 7;
};

class SpatialPooler {
 public:
  explicit SpatialPooler(const SpatialPoolerParams& p);

  SDR activate(const SDR& input);
  SDR activate(const SDR& input, const SDR& feedback);
  const std::vector<float>& boosts() const { return boost_; }
  double mean_overlap() const { return last_mean_overlap_; }
  size_t dead_columns() const;
  size_t probe_dead(const std::vector<SDR>& probes) const;
  size_t mean_synapses() const;
  size_t prune(float min_perm);

  void reset_homeostasis();
  void set_tie_jitter(float j) { p_.tie_jitter = j; }
  float tie_jitter() const { return static_cast<float>(p_.tie_jitter); }

  size_t input_size() const { return p_.input_size; }
  size_t column_count() const { return p_.column_count; }

 private:
  float overlap_of(size_t col, const SDR& in) const;
  SDR activate_impl(const SDR& input, const SDR* feedback);
  void learn(size_t col, const SDR& in);
  void homeostat(size_t col);

  SpatialPoolerParams p_;
  std::vector<std::vector<Syn>> syn_;
  std::vector<float> boost_;
  std::vector<float> rate_;
  double last_mean_overlap_ = 0.0;
  Rng rng_;
};

}  // namespace bm