#include "bm/temporal_memory.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace bm {

TemporalMemory::TemporalMemory(const TemporalMemoryParams& p)
    : p_(p), cells_(p.column_count * p.cells_per_column), rng_(p.seed) {
  segs_.assign(cells_, {});
  prev_active_ = SDR(cells_, 0);
  last_predictive_ = SDR(cells_, 0);
  prev_predictive_ = SDR(cells_, 0);
  for (size_t c = 0; c < p.column_count; ++c) {
    for (size_t i = 0; i < p.initial_segments; ++i) {
      uint32_t cell = static_cast<uint32_t>(c * p.cells_per_column + rng_.below(static_cast<uint32_t>(p.cells_per_column)));
      Segment s;
      size_t syn = p.new_segment_min + rng_.below(4);
      std::vector<uint32_t> pool;
      for (uint32_t x = 0; x < cells_; ++x) {
        if (column_of(x) != c) pool.push_back(x);
      }
      std::vector<uint32_t> shuffled = pool;
      for (size_t i = shuffled.size(); i > 1; --i) {
        size_t j = rng_.below(static_cast<uint32_t>(i));
        std::swap(shuffled[i - 1], shuffled[j]);
      }
      for (size_t i = 0; i < syn && i < shuffled.size(); ++i) {
        s.cells.push_back(shuffled[i]);
        s.perms.push_back(static_cast<uint8_t>(p_.permanence_threshold + rng_.below(20)));
      }
      segs_[cell].push_back(std::move(s));
    }
  }
}

size_t TemporalMemory::active_synapses(const Segment& s, const SDR& prev) const {
  size_t n = 0;
  for (size_t i = 0; i < s.cells.size(); ++i) {
    if (s.perms[i] >= p_.permanence_threshold && prev.test(s.cells[i])) ++n;
  }
  return n;
}

void TemporalMemory::adapt_segment(Segment& s, const SDR& prev, float polarity) {
  for (size_t i = 0; i < s.cells.size(); ++i) {
    uint8_t& w = s.perms[i];
    if (prev.test(s.cells[i])) {
      int step = static_cast<int>(std::lround((polarity > 0 ? p_.permanence_inc : -p_.permanence_dec) * learn_gain_));
      int v = static_cast<int>(w) + step;
      w = static_cast<uint8_t>(std::min(255, std::max(0, v)));
    } else if (polarity < 0) {
      int v = static_cast<int>(w) - static_cast<int>(std::lround(p_.permanence_dec * learn_gain_));
      w = static_cast<uint8_t>(std::max(0, v));
    }
  }
  if (s.age < 255) ++s.age;
}

void TemporalMemory::grow_segment(uint32_t cell, const std::vector<uint32_t>& idx) {
  if (idx.empty()) return;
  if (segs_[cell].size() >= p_.max_segments_per_cell) segs_[cell].erase(segs_[cell].begin());
  Segment s;
  size_t take = std::min<size_t>(idx.size(), 8);
  size_t start = rng_.below(static_cast<uint32_t>(std::max<size_t>(1, idx.size() - take + 1)));
  for (size_t i = 0; i < take; ++i) {
    s.cells.push_back(idx[start + i]);
    s.perms.push_back(static_cast<uint8_t>(p_.permanence_threshold + 4));
  }
  segs_[cell].push_back(std::move(s));
}

TMStep TemporalMemory::step(const SDR& active_columns) {
  TMStep out{SDR(cells_, 0), SDR(cells_, 0), 0, 0};
  SDR context = prev_active_;
  if (p_.use_predictive_in_prev) context |= prev_predictive_;
  const SDR prev = context;
  const std::vector<uint32_t> prev_idx = prev.indices();

  std::vector<uint32_t> winners;
  std::vector<uint32_t> predictive_cells;
  std::vector<uint32_t> burst_columns;

  for (uint32_t col : active_columns.indices()) {
    if (col >= p_.column_count) continue;
    size_t base = static_cast<size_t>(col) * p_.cells_per_column;
    int best_cell = -1;
    size_t best_score = 0;
    bool any_predictive = false;
    for (size_t i = 0; i < p_.cells_per_column; ++i) {
      const std::vector<Segment>& segs = segs_[base + i];
      size_t score = 0;
      for (const Segment& s : segs) {
        size_t a = active_synapses(s, prev);
        if (a >= static_cast<size_t>(p_.activation_threshold)) ++score;
      }
      if (score > 0) {
        any_predictive = true;
        predictive_cells.push_back(static_cast<uint32_t>(base + i));
        if (best_cell < 0 || score > best_score) {
          best_cell = static_cast<int>(base + i);
          best_score = score;
        }
      }
    }
    if (any_predictive && best_cell >= 0) {
      winners.push_back(static_cast<uint32_t>(best_cell));
      out.predictive.set(static_cast<uint32_t>(best_cell));
    } else {
      burst_columns.push_back(col);
      for (size_t i = 0; i < p_.cells_per_column; ++i) {
        winners.push_back(static_cast<uint32_t>(base + i));
      }
    }
  }

  for (uint32_t c : winners) out.active.set(c);

  for (size_t col : burst_columns) {
    size_t base = col * p_.cells_per_column;
    for (size_t i = 0; i < p_.cells_per_column; ++i) {
      uint32_t cell = static_cast<uint32_t>(base + i);
      bool grew = false;
      for (Segment& s : segs_[cell]) {
        if (active_synapses(s, prev) > 0) {
          adapt_segment(s, prev, +1.0f);
          grew = true;
        }
      }
      if (!grew) grow_segment(cell, prev_idx);
    }
  }

  for (uint32_t cell : predictive_cells) {
    if (out.active.test(cell)) continue;
    for (Segment& s : segs_[cell]) {
      if (active_synapses(s, prev) > 0) adapt_segment(s, prev, -1.0f);
    }
  }

  for (uint32_t cell : winners) {
    for (const Segment& s : segs_[cell]) {
      if (active_synapses(s, prev) > 0) break;
    }
  }

  size_t predicted_hits = 0;
  for (uint32_t c : out.active.indices()) {
    if (out.predictive.test(c)) ++predicted_hits;
  }
  size_t burst_cells = burst_columns.size() * p_.cells_per_column;
  out.bursting_columns = burst_columns.size();
  out.predicted_active = predicted_hits;
  size_t active_cols = active_columns.count();
  last_anomaly_ = active_cols > 0
                      ? static_cast<double>(burst_columns.size()) / static_cast<double>(active_cols)
                      : 0.0;
  (void)burst_cells;
  last_accuracy_ = out.active.count() > 0
                       ? static_cast<double>(predicted_hits) / static_cast<double>(out.active.count())
                       : 0.0;

  prev_active_ = out.active;
  prev_predictive_ = out.predictive;
  last_predictive_ = out.predictive;
  return out;
}

size_t TemporalMemory::segment_count() const {
  size_t total = 0;
  for (const auto& v : segs_) total += v.size();
  return total;
}

void TemporalMemory::reset() {
  prev_active_ = SDR(cells_, 0);
  prev_predictive_ = SDR(cells_, 0);
  last_predictive_ = SDR(cells_, 0);
  last_anomaly_ = 0.0;
  last_accuracy_ = 0.0;
}

}  // namespace bm