#include "bm/cleanup.hpp"

#include <algorithm>
#include <limits>

namespace bm {

CleanupMemory::CleanupMemory(size_t n, size_t max_items)
    : n_(n), words_((n + 63) / 64), max_items_(max_items) {
  items_.reserve(max_items_ * words_);
  probe_.assign(words_, 0);
}

void CleanupMemory::add(const SDR& code) {
  if (code.size() != n_ || items_.size() / words_ >= max_items_) return;
  size_t k = items_.size() / words_;
  items_.resize((k + 1) * words_, 0);
  uint64_t* dst = items_.data() + k * words_;
  for (uint32_t i : code.indices()) {
    if (i < n_) dst[static_cast<size_t>(i) >> 6] |= (uint64_t{1} << (i & 63));
  }
}

std::pair<size_t, size_t> CleanupMemory::nearest(const SDR& code) const {
  if (items_.empty()) return {0, 0};
  const size_t count = items_.size() / words_;
  size_t best = 0;
  size_t best_ov = 0;
  std::fill(probe_.begin(), probe_.end(), 0);
  for (uint32_t i : code.indices()) {
    if (i < n_) probe_[static_cast<size_t>(i) >> 6] |= (uint64_t{1} << (i & 63));
  }
  const uint64_t* base = items_.data();
  const uint64_t* probe = probe_.data();
  const size_t words = words_;
  for (size_t k = 0; k < count; ++k) {
    const uint64_t* it = base + k * words;
    size_t ov = 0;
    for (size_t w = 0; w < words; ++w)
      ov += static_cast<size_t>(__builtin_popcountll(it[w] & probe[w]));
    if (ov > best_ov) {
      best_ov = ov;
      best = k + 1;
    }
  }
  return {best ? best - 1 : 0, best_ov};
}

SDR CleanupMemory::clean(const SDR& noisy, float min_overlap) const {
  auto [idx, ov] = nearest(noisy);
  size_t need = static_cast<size_t>(min_overlap * static_cast<float>(noisy.count()));
  if (items_.empty() || ov < need) return noisy;
  SDR out(n_, 0);
  const uint64_t* row = items_.data() + idx * words_;
  for (size_t w = 0; w < words_; ++w) {
    uint64_t bits = row[w];
    while (bits) {
      uint32_t b = static_cast<uint32_t>(__builtin_ctzll(bits));
      out.set(static_cast<uint32_t>(w * 64 + b));
      bits &= bits - 1;
    }
  }
  return out;
}

bool CleanupMemory::is_known(const SDR& code, float min_overlap) const {
  auto [idx, ov] = nearest(code);
  (void)idx;
  size_t need = static_cast<size_t>(min_overlap * static_cast<float>(code.count()));
  return !items_.empty() && ov >= need;
}

}  // namespace bm