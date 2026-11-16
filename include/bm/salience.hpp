#pragma once

namespace bm {

struct SalienceParams {
  float lambda = 0.2f;
  float base_gain = 1.0f;
  float max_gain = 4.0f;
  float uncertainty_tau = 50.0f;
};

class Salience {
 public:
  explicit Salience(const SalienceParams& p = {}) : p_(p) {}

  void update(float anomaly);
  float gain() const { return gain_; }
  float learning_rate(float base_lr) const { return base_lr * gain_; }
  float anomaly() const { return ema_; }
  float uncertainty() const { return uncertainty_; }

 private:
  SalienceParams p_;
  float ema_ = 0.0f;
  float var_ = 0.0f;
  float uncertainty_ = 0.0f;
  float gain_ = 1.0f;
};

}  // namespace bm