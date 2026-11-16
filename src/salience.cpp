#include "bm/salience.hpp"

#include <algorithm>
#include <cmath>

namespace bm {

void Salience::update(float anomaly) {
  float a = std::max(0.0f, std::min(1.0f, anomaly));
  ema_ += p_.lambda * (a - ema_);
  float diff = a - ema_;
  var_ = (1.0f - p_.lambda) * var_ + p_.lambda * diff * diff;
  uncertainty_ += (std::sqrt(var_) - uncertainty_) / p_.uncertainty_tau;

  float target = p_.base_gain * (1.0f + 2.0f * ema_ + uncertainty_);
  target = std::min(p_.max_gain, std::max(p_.base_gain, target));
  gain_ += 0.2f * (target - gain_);
}

}  // namespace bm