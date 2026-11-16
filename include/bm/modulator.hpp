#pragma once

#include <cstdint>
#include <unordered_map>

namespace bm {

struct Modulators {
  float rpe = 0.0f;
  float novelty = 0.0f;
  float ach = 1.0f;
  float na = 1.0f;

  float gain() const { return ach * (1.0f + 0.5f * na); }
  float learning_scale() const { return gain(); }
};

struct ModulatorParams {
  float alpha_v = 0.1f;
  float gamma = 0.9f;
  float novelty_lambda = 0.2f;
  float ach_base = 1.0f;
  float ach_k = 0.8f;
  float na_theta = 0.3f;
  float na_gain = 2.0f;
  float na_decay = 0.9f;
  float uncertainty_tau = 100.0f;
  float arousal_bias = 0.0f;
  float valence_bias = 0.0f;
};

class ModulatorSystem {
 public:
  explicit ModulatorSystem(const ModulatorParams& p = {}) : p_(p) {}

  void observe_state(uint64_t state);
  void observe_reward(float reward);
  void observe_anomaly(float anomaly);
  void set_interoception(float arousal, float valence);
  void tick();

  const Modulators& modulators() const { return m_; }
  float value() const { return value_; }
  float uncertainty() const { return uncertainty_; }
  float predicted_reward() const { return value_; }

 private:
  ModulatorParams p_;
  Modulators m_{};
  std::unordered_map<uint64_t, float> values_;
  uint64_t state_ = 0;
  float value_ = 0.0f;
  float anomaly_ema_ = 0.0f;
  float anomaly_var_ = 0.0f;
  float uncertainty_ = 0.0f;
  float ach_ = 1.0f;
  float na_ = 1.0f;
  float novelty_ = 0.0f;
};

}  // namespace bm