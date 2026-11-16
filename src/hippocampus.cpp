#include "bm/hippocampus.hpp"

#include <algorithm>
#include <numeric>

namespace bm {

Hippocampus::Hippocampus(size_t input_size, size_t dentate_size, size_t k, size_t fan_in,
                         uint64_t seed)
    : input_size_(input_size), dentate_size_(dentate_size), k_(k), fan_in_(fan_in) {
  proj_.assign(dentate_size_, {});
  Rng rng(seed);
  for (size_t d = 0; d < dentate_size_; ++d) {
    std::vector<uint8_t> taken(input_size_, 0);
    proj_[d].reserve(fan_in_);
    while (proj_[d].size() < fan_in_) {
      uint32_t idx = static_cast<uint32_t>((rng.next() >> 32) % input_size_);
      if (taken[idx]) continue;
      taken[idx] = 1;
      proj_[d].push_back(idx);
    }
  }
  store_.reserve(1024);
}

SDR Hippocampus::dentate(const SDR& cortical) const {
  std::vector<float> score(dentate_size_, 0.0f);
  for (size_t d = 0; d < dentate_size_; ++d) {
    float s = 0.0f;
    for (uint32_t idx : proj_[d]) {
      if (cortical.test(idx)) s += 1.0f;
    }
    score[d] = s;
  }
  std::vector<size_t> order(dentate_size_);
  std::iota(order.begin(), order.end(), 0);
  size_t k = std::min(k_, dentate_size_);
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](size_t a, size_t b) { return score[a] > score[b]; });
  SDR out(dentate_size_, 0);
  for (size_t i = 0; i < k; ++i) out.set(static_cast<uint32_t>(order[i]));
  return out;
}

void Hippocampus::store(const SDR& cortical) {
  SDR code = dentate(cortical);
  for (const Entry& e : store_) {
    if (e.dentate == code) return;
  }
  if (store_.size() >= store_.capacity()) store_.erase(store_.begin());
  store_.push_back(Entry{cortical, code});
}

SDR Hippocampus::complete(const SDR& cue, float min_overlap) const {
  if (store_.empty() || cue.empty()) return SDR();
  const Entry* best = nullptr;
  double best_score = 0.0;
  for (const Entry& e : store_) {
    double score = static_cast<double>(e.cortical.overlap(cue)) / static_cast<double>(cue.count());
    if (score > best_score) {
      best_score = score;
      best = &e;
    }
  }
  if (best == nullptr || best_score < min_overlap) return SDR();
  return best->dentate;
}

float Hippocampus::mismatch(const SDR& a, const SDR& b) const {
  if (a.empty() || b.empty()) return 1.0f;
  return 1.0f - a.jaccard(b);
}

}  // namespace bm