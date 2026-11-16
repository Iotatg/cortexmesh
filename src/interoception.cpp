#include "bm/interoception.hpp"

#include <algorithm>

namespace bm {

Interoception::Interoception(size_t n, size_t per_channel)
    : n_(n),
      per_channel_(per_channel),
      hr_(per_channel_, per_channel_ / 4, 40.0, 160.0, 30),
      hrv_(per_channel_, per_channel_ / 4, 5.0, 150.0, 30),
      breath_(per_channel_, per_channel_ / 4, 5.0, 30.0, 30),
      effort_(per_channel_, per_channel_ / 4, 0.0, 1.0, 20) {}

SDR Interoception::encode(const BodyState& b) const {
  SDR out(n_, 0);
  auto place = [&](const SDR& part, size_t offset) {
    for (uint32_t idx : part.indices()) {
      size_t bit = offset + (idx % per_channel_);
      if (bit < n_) out.set(static_cast<uint32_t>(bit));
    }
  };
  place(hr_.encode(b.heart_rate), 0);
  place(hrv_.encode(b.hrv), per_channel_);
  place(breath_.encode(b.breathing), per_channel_ * 2);
  place(effort_.encode(b.effort), per_channel_ * 3);
  return out;
}

float Interoception::arousal(const BodyState& b) const {
  float hr = std::max(0.0f, std::min(1.0f, (b.heart_rate - 50.0f) / 90.0f));
  float br = std::max(0.0f, std::min(1.0f, (b.breathing - 8.0f) / 18.0f));
  return 0.7f * hr + 0.3f * br;
}

float Interoception::valence(const BodyState& b) const {
  return std::max(0.0f, std::min(1.0f, (b.hrv - 20.0f) / 90.0f));
}

}  // namespace bm