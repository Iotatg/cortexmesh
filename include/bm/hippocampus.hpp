#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "bm/encoder.hpp"
#include "bm/sdr.hpp"

namespace bm {

class Hippocampus {
 public:
  Hippocampus(size_t input_size = 1024, size_t dentate_size = 2048, size_t k = 30,
              size_t fan_in = 16, uint64_t seed = 17);

  SDR dentate(const SDR& cortical) const;
  void store(const SDR& cortical);
  SDR complete(const SDR& cue, float min_overlap = 0.5f) const;
  float mismatch(const SDR& a, const SDR& b) const;
  size_t stored() const { return store_.size(); }
  size_t capacity() const { return store_.capacity(); }

 private:
  struct Entry {
    SDR cortical;
    SDR dentate;
  };

  std::vector<Entry> store_;

 private:
  size_t input_size_, dentate_size_, k_, fan_in_;
  std::vector<std::vector<uint32_t>> proj_;
};

}  // namespace bm