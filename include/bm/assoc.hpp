#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "bm/sdr.hpp"

namespace bm {

class AssociativeMemory {
 public:
  AssociativeMemory(size_t in, size_t out, size_t k_out, float lr = 0.05f);

  void learn(const SDR& in, const SDR& out);
  void learn_batch(const std::vector<std::pair<SDR, SDR>>& pairs);
  SDR recall(const SDR& in) const;
  SDR recall(const SDR& in, size_t k) const;

  std::vector<std::pair<uint32_t, float>> scores(const SDR& in) const;
  float weight(size_t i, size_t j) const { return w_[i * out_ + j]; }

  size_t in_size() const { return in_; }
  size_t out_size() const { return out_; }
  size_t k() const { return k_; }

 private:
  size_t in_, out_, k_;
  float lr_;
  std::vector<float> w_;
};

}  // namespace bm