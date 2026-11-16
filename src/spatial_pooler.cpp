#include "bm/spatial_pooler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace bm {

SpatialPooler::SpatialPooler(const SpatialPoolerParams& p) : p_(p), rng_(p.seed) {
  syn_.assign(p_.column_count, {});
  boost_.assign(p_.column_count, 1.0f);
  rate_.assign(p_.column_count, 0.0f);

  double permille = std::min(1000.0, std::max(1.0, p_.connected_permille));
  size_t target = std::max<size_t>(1, static_cast<size_t>(p_.synapses_per_column * permille / 1000.0));
  uint8_t lo = p_.permanence_init_min;
  uint8_t hi = std::max<uint8_t>(p_.permanence_init_max, static_cast<uint8_t>(lo + 1));
  for (size_t c = 0; c < p_.column_count; ++c) {
    std::vector<Syn>& s = syn_[c];
    s.reserve(target);
    std::vector<uint8_t> taken(p_.input_size, 0);
    while (s.size() < target) {
      uint32_t idx = static_cast<uint32_t>((rng_.next() >> 32) % p_.input_size);
      if (taken[idx]) continue;
      taken[idx] = 1;
      s.push_back(Syn{idx, static_cast<uint8_t>(lo + rng_.below(hi - lo + 1))});
    }
  }
}

float SpatialPooler::overlap_of(size_t col, const SDR& in) const {
  float total = 0.0f;
  for (const Syn& s : syn_[col]) {
    if (s.w >= p_.permanence_threshold && in.test(s.idx)) total += static_cast<float>(s.w) / 255.0f;
  }
  return total;
}

void SpatialPooler::learn(size_t col, const SDR& in) {
  for (Syn& s : syn_[col]) {
    if (in.test(s.idx)) {
      int v = static_cast<int>(s.w) + p_.permanence_inc;
      s.w = static_cast<uint8_t>(std::min(255, v));
    } else {
      int v = static_cast<int>(s.w) - p_.permanence_dec;
      s.w = (v <= 0) ? 0 : static_cast<uint8_t>(v);
    }
  }
  homeostat(col);
  size_t growth_floor = p_.growth_min_size ? p_.growth_min_size : p_.synapses_per_column / 3;
  if (syn_[col].size() < growth_floor && in.count() > 0 &&
      static_cast<size_t>((rng_.next() >> 32) % 100) < static_cast<size_t>(p_.growth_prob)) {
    for (int attempt = 0; attempt < 8; ++attempt) {
      uint32_t idx = static_cast<uint32_t>(rng_.below(static_cast<uint32_t>(p_.input_size)));
      if (!in.test(idx)) continue;
      bool present = false;
      for (const Syn& s : syn_[col]) {
        if (s.idx == idx) {
          present = true;
          break;
        }
      }
      if (!present) syn_[col].push_back(Syn{idx, p_.permanence_init_min});
      break;
    }
  }
}

void SpatialPooler::homeostat(size_t col) {
  double sum = 0.0;
  for (const Syn& s : syn_[col]) sum += s.w;
  size_t nominal = std::max<size_t>(
      1, static_cast<size_t>(p_.synapses_per_column * p_.connected_permille / 1000.0));
  double target = static_cast<double>(nominal) *
                  (static_cast<double>(p_.permanence_init_min + p_.permanence_init_max) / 2.0);
  if (sum <= 0.0 || target <= 0.0) return;
  double scale = target / sum;
  if (scale < 0.98 || scale > 1.02) {
    for (Syn& s : syn_[col]) {
      int v = static_cast<int>(std::lround(static_cast<double>(s.w) * scale));
      s.w = static_cast<uint8_t>(std::max(0, std::min(255, v)));
    }
  }
}

SDR SpatialPooler::activate(const SDR& input) { return activate_impl(input, nullptr); }

SDR SpatialPooler::activate(const SDR& input, const SDR& feedback) {
  return activate_impl(input, &feedback);
}

SDR SpatialPooler::activate_impl(const SDR& input, const SDR* feedback) {
  std::vector<float> overlap(p_.column_count, 0.0f);
  double sum = 0.0;
  size_t counted = 0;
  for (size_t c = 0; c < p_.column_count; ++c) {
    float total = overlap_of(c, input);
    if (total > 0.0f) {
      sum += total;
      ++counted;
    }
    float jitter = 1.0f + static_cast<float>(rng_.below(1000)) * (static_cast<float>(p_.tie_jitter) / 1000.0f);
    overlap[c] = total * boost_[c] * jitter;
    if (feedback != nullptr && feedback->test(static_cast<uint32_t>(c))) {
      overlap[c] *= static_cast<float>(1.0 + p_.feedback_gain);
    }
  }
  last_mean_overlap_ = counted ? sum / static_cast<double>(counted) : 0.0;

  size_t k = std::max<size_t>(1, std::min(p_.column_count, p_.active_columns));
  std::vector<size_t> order(p_.column_count);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](size_t a, size_t b) { return overlap[a] > overlap[b]; });

  SDR out(p_.column_count, 0);
  for (size_t i = 0; i < k; ++i) {
    size_t c = order[i];
    if (overlap[c] <= 0.0f) break;
    out.set(static_cast<uint32_t>(c));
    learn(c, input);
    boost_[c] = std::max(1.0f, boost_[c] - static_cast<float>(p_.boost_decay));
    rate_[c] += 1.0f;
  }

  for (size_t c = 0; c < p_.column_count; ++c) {
    rate_[c] *= 0.999f;
    if (rate_[c] < 0.01f) {
      boost_[c] = std::min(static_cast<float>(p_.boost_max), boost_[c] + static_cast<float>(p_.boost_inc));
    }
  }
  return out;
}

size_t SpatialPooler::dead_columns() const {
  size_t dead = 0;
  for (float r : rate_) {
    if (r < 0.002f) ++dead;
  }
  return dead;
}

size_t SpatialPooler::probe_dead(const std::vector<SDR>& probes) const {
  size_t dead = 0;
  for (size_t c = 0; c < p_.column_count; ++c) {
    bool ever = false;
    for (const SDR& pr : probes) {
      if (overlap_of(c, pr) > 0.0f) {
        ever = true;
        break;
      }
    }
    if (!ever) ++dead;
  }
  return dead;
}

size_t SpatialPooler::prune(float min_perm) {
  size_t nominal = std::max<size_t>(
      1, static_cast<size_t>(p_.synapses_per_column * p_.connected_permille / 1000.0));
  size_t keep_min = std::max<size_t>(8, nominal / 2);
  size_t removed = 0;
  for (size_t c = 0; c < syn_.size(); ++c) {
    std::vector<Syn>& col = syn_[c];
    if (col.size() <= keep_min) continue;
    std::vector<size_t> order(col.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return col[a].w > col[b].w;
    });
    std::vector<uint8_t> drop(col.size(), 0);
    for (size_t rank = keep_min; rank < order.size(); ++rank) {
      if (static_cast<float>(col[order[rank]].w) < min_perm) drop[order[rank]] = 1;
    }
    std::vector<Syn> kept;
    kept.reserve(col.size());
    for (size_t i = 0; i < col.size(); ++i) {
      if (drop[i]) {
        ++removed;
      } else {
        kept.push_back(col[i]);
      }
    }
    col.swap(kept);
  }
  return removed;
}

size_t SpatialPooler::mean_synapses() const {
  size_t total = 0;
  for (const auto& s : syn_) total += s.size();
  return syn_.empty() ? 0 : total / syn_.size();
}

void SpatialPooler::reset_homeostasis() {
  std::fill(boost_.begin(), boost_.end(), 1.0f);
  std::fill(rate_.begin(), rate_.end(), 0.0f);
}

}  // namespace bm