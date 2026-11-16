#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "bm/encoder.hpp"
#include "bm/hippocampus.hpp"
#include "bm/sdr.hpp"
#include "bm/valuation.hpp"

namespace bm {

struct Decision {
  SDR action;
  float value = 0.0f;
  float model_free_value = 0.0f;
  float model_based_value = 0.0f;
  bool model_based = false;
  float confidence = 0.0f;
};

struct DualControllerParams {
  size_t state_size = 1024;
  size_t action_size = 1024;
  size_t code_size = 2048;
  size_t code_k = 24;
  size_t dentate_size = 2048;
  size_t dentate_k = 30;
  float cost = 0.05f;
  float uncertainty_threshold = 0.5f;
  uint64_t seed = 5;
};

class DualController {
 public:
  explicit DualController(const DualControllerParams& p = {});

  void learn_model_free(const SDR& state, const SDR& action, float reward);
  void learn_model_based(const SDR& state, const SDR& action, const SDR& outcome, float outcome_reward);
  void tick();

  Decision decide(const SDR& state, const std::vector<SDR>& actions, float uncertainty);

  float model_free_value(const SDR& state, const SDR& action) const;
  float model_based_value(const SDR& state, const SDR& action) const;

 private:
  uint64_t key_of(const SDR& state, const SDR& action) const;

  DualControllerParams p_;
  Encoder enc_;
  mutable std::unordered_map<uint64_t, Valuation> action_values_;
  Valuation outcome_value_;
  Hippocampus hp_;
  std::unordered_map<uint64_t, SDR> outcomes_;
};

}  // namespace bm