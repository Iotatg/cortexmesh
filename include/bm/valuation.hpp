#pragma once

#include <cstdint>
#include <vector>

#include "bm/sdr.hpp"

namespace bm {

class Valuation {
 public:
  explicit Valuation(size_t n = 4096, float alpha = 0.05f, float trace_decay = 0.85f);

  float predict(const SDR& state) const;
  void present(const SDR& state);
  void reward(float r);
  void tick();

  float rpe() const { return last_rpe_; }
  float trace_mass() const;
  size_t size() const { return w_.size(); }
  float weight(size_t i) const { return w_[i]; }

 private:
  std::vector<float> w_;
  std::vector<float> trace_;
  float alpha_;
  float trace_decay_;
  float last_rpe_ = 0.0f;
  float baseline_ = 0.0f;
  bool has_input_ = false;
};

}  // namespace bm