#include "bm/assoc.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace bm {

AssociativeMemory::AssociativeMemory(size_t in, size_t out, size_t k_out, float lr)
    : in_(in), out_(out), k_(std::max<size_t>(1, std::min(k_out, out))),
      lr_(lr), w_(in * out, 0.0f) {}

void AssociativeMemory::learn(const SDR& in, const SDR& out) {
  if (in.size() != in_ || out.size() != out_) return;
  // gentle decay keeps frequent patterns from swamping rare ones
  for (auto& x : w_) x *= 0.99999f;

  std::vector<uint32_t> ins = in.indices();
  std::vector<uint32_t> outs = out.indices();
  for (uint32_t j : outs) {
    float* col = w_.data() + j;
    for (uint32_t i : ins) col[i * out_] += lr_;
  }
}

void AssociativeMemory::learn_batch(const std::vector<std::pair<SDR, SDR>>& pairs) {
  // one decay for the whole batch, then accumulate: same fixed point as repeated
  // learn() calls, without paying the O(weights) decay per pair
  for (auto& x : w_) x *= 0.99999f;
  for (const auto& [in, out] : pairs) {
    if (in.size() != in_ || out.size() != out_) continue;
    for (uint32_t j : out.indices()) {
      float* col = w_.data() + j;
      for (uint32_t i : in.indices()) col[i * out_] += lr_;
    }
  }
}

std::vector<std::pair<uint32_t, float>> AssociativeMemory::scores(const SDR& in) const {
  std::vector<float> s(out_, 0.0f);
  for (uint32_t i : in.indices()) {
    const float* row = w_.data() + i * out_;
    for (size_t j = 0; j < out_; ++j) s[j] += row[j];
  }
  std::vector<std::pair<uint32_t, float>> ranked;
  ranked.reserve(out_);
  for (size_t j = 0; j < out_; ++j) ranked.emplace_back(static_cast<uint32_t>(j), s[j]);
  std::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  return ranked;
}

SDR AssociativeMemory::recall(const SDR& in) const { return recall(in, k_); }

SDR AssociativeMemory::recall(const SDR& in, size_t k) const {
  auto ranked = scores(in);
  SDR r(out_, 0);
  for (size_t i = 0; i < k && i < ranked.size(); ++i) r.set(ranked[i].first);
  return r;
}

}  // namespace bm