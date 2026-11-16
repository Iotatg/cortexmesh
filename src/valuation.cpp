#include "bm/valuation.hpp"

#include <algorithm>
#include <cmath>

namespace bm {

Valuation::Valuation(size_t n, float alpha, float trace_decay)
    : w_(n, 0.0f), trace_(n, 0.0f), alpha_(alpha), trace_decay_(trace_decay) {}

float Valuation::predict(const SDR& state) const {
  if (state.empty()) return 0.0f;
  double sum = 0.0;
  for (uint32_t i : state.indices()) sum += w_[i];
  return static_cast<float>(sum / static_cast<double>(state.count()) - baseline_);
}

void Valuation::present(const SDR& state) {
  for (float& t : trace_) t *= trace_decay_;
  for (uint32_t i : state.indices()) trace_[i] = 1.0f;
  has_input_ = true;
}

void Valuation::reward(float r) {
  if (!has_input_) return;
  double sum = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < trace_.size(); ++i) {
    if (trace_[i] > 0.5f) {
      sum += w_[i];
      ++count;
    }
  }
  float predicted = count ? static_cast<float>(sum / static_cast<double>(count) - baseline_) : 0.0f;
  last_rpe_ = r - predicted;

  for (size_t i = 0; i < trace_.size(); ++i) {
    if (trace_[i] <= 0.0f) continue;
    float delta = alpha_ * last_rpe_ * trace_[i];
    w_[i] = std::max(-10.0f, std::min(10.0f, w_[i] + delta));
  }
}

void Valuation::tick() {
  double sum = 0.0;
  for (size_t i = 0; i < w_.size(); ++i) {
    w_[i] *= 0.9995f;
    if (std::fabs(w_[i]) < 1e-4f) w_[i] = 0.0f;
    sum += w_[i];
  }
  float mean = w_.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(w_.size()));
  baseline_ += 0.05f * (mean - baseline_);
}

float Valuation::trace_mass() const {
  float sum = 0.0f;
  for (float t : trace_) sum += t;
  return sum;
}

}  // namespace bm