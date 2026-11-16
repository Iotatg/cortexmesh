#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bm/sdr.hpp"

namespace bm {

// Cleanup memory (Kanerva): denoise an approximate sparse code by snapping it to the
// closest stored canonical code.
//
// The associative read-out returns an approximate code, not the exact one. Re-using that
// code as the next cue compounds the error (measured: 5-hop chains reach 0.00 even though
// per-hop recall is 0.90). Cleanup breaks the loop: the retrieved code is matched against
// the set of known codes and replaced by the canonical one before it is used again.
//
// Storage is a plain bitset per item, so a clean is a popcount over the set: no training,
// no decay, and the state size is fixed at max_items * n / 8 bytes.
class CleanupMemory {
 public:
  CleanupMemory(size_t n, size_t max_items);

  void add(const SDR& code);
  void clear() { items_.clear(); }

  // Returns (index, overlap) of the closest stored code, overlap in bits.
  std::pair<size_t, size_t> nearest(const SDR& code) const;

  // Canonical code for `code`, or `code` itself when nothing is close enough.
  SDR clean(const SDR& noisy, float min_overlap = 0.6f) const;

  // True when a stored code is close enough to be trusted.
  bool is_known(const SDR& code, float min_overlap = 0.6f) const;

  size_t size() const { return items_.size() / words_; }
  size_t capacity() const { return max_items_; }
  size_t width() const { return n_; }
  size_t bytes() const { return items_.capacity() * sizeof(uint64_t); }

 private:
  mutable std::vector<uint64_t> probe_;
  size_t n_ = 0;
  size_t words_ = 0;
  size_t max_items_ = 0;
  std::vector<uint64_t> items_;  // flat: item k lives at k * words_
};

}  // namespace bm