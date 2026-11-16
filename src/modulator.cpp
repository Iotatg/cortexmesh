#include "bm/modulator.hpp"

#include <algorithm>
#include <cmath>

namespace bm {

void ModulatorSystem::observe_state(uint64_t state) {
  state_ = state;
  auto it = values_.find(state);
  value_ = (it == values_.end()) ? 0.0f : it->second;
  m_.rpe = 0.0f;
}

void ModulatorSystem::observe_reward(float reward) {
  float rpe = reward - value_;
  m_.rpe = rpe;
  value_ += p_.alpha_v * rpe;
  values_[state_] = value_;
}

void ModulatorSystem::observe_anomaly(float anomaly) {
  float a = std::max(0.0f, std::min(1.0f, anomaly));
  float prev = anomaly_ema_;
  anomaly_ema_ = prev + p_.novelty_lambda * (a - prev);
  anomaly_var_ = anomaly_var_ * 0.98f + (a - anomaly_ema_) * (a - anomaly_ema_) * 0.02f;

  float mean_sq = anomaly_var_ + anomaly_ema_ * anomaly_ema_;
  uncertainty_ = std::sqrt(std::max(0.0f, mean_sq));

  novelty_ = (1.0f - p_.novelty_lambda) * novelty_ + p_.novelty_lambda * a;
  m_.novelty = novelty_;

  float ach_target = p_.ach_base + p_.ach_k * uncertainty_ + p_.arousal_bias * a;
  ach_ += 0.1f * (ach_target - ach_);
  m_.ach = std::max(0.1f, ach_);

  float na_target = (novelty_ > p_.na_theta) ? 1.0f + p_.na_gain * (novelty_ - p_.na_theta) : 1.0f;
  na_ = na_target + (na_ - na_target) * p_.na_decay;
  m_.na = std::max(0.5f, na_);
}

void ModulatorSystem::set_interoception(float arousal, float valence) {
  p_.arousal_bias = std::max(-1.0f, std::min(1.0f, arousal));
  p_.valence_bias = std::max(-1.0f, std::min(1.0f, valence));
}

void ModulatorSystem::tick() {
  novelty_ *= 0.995f;
  na_ = 1.0f + (na_ - 1.0f) * p_.na_decay;
  m_.novelty = novelty_;
  m_.na = na_;
  m_.ach = ach_;
}

}  // namespace bm