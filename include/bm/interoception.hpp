#pragma once

#include "bm/encoder.hpp"
#include "bm/sdr.hpp"

namespace bm {

struct BodyState {
  float heart_rate = 70.0f;
  float hrv = 50.0f;
  float breathing = 14.0f;
  float effort = 0.0f;
};

class Interoception {
 public:
  explicit Interoception(size_t n = 512, size_t per_channel = 64);

  SDR encode(const BodyState& b) const;
  float arousal(const BodyState& b) const;
  float valence(const BodyState& b) const;
  size_t size() const { return n_; }

 private:
  size_t n_, per_channel_;
  ScalarEncoder hr_, hrv_, breath_, effort_;
};

}  // namespace bm