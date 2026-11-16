#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "bm/encoder.hpp"
#include "bm/sdr.hpp"

namespace bm {

struct WordEntry {
  std::vector<std::string> phones;
  std::vector<std::string> concepts;
  float valence = 0.0f;
  float arousal = 0.2f;
};

class WordEncoder {
 public:
  explicit WordEncoder(size_t n = 512, size_t k = 24, uint64_t seed = 11);

  SDR phone(const std::string& p) const;
  SDR concept_code(const std::string& c) const;
  SDR encode(const std::string& word) const;
  SDR valence_code(float v) const;
  SDR arousal_code(float v) const;

  const WordEntry* entry(const std::string& word) const;
  bool known(const std::string& word) const { return entry(word) != nullptr; }
  const std::vector<std::string>& vocabulary() const { return vocab_; }

  size_t size() const { return n_; }
  size_t k() const { return k_; }

 private:
  SDR feature_code(size_t slice_index, size_t slice_count, size_t value, size_t max_value,
                    size_t width) const;

  size_t n_, k_;
  uint64_t seed_;
  mutable Encoder enc_;
  std::unordered_map<std::string, WordEntry> words_;
  std::vector<std::string> vocab_;
  mutable std::unordered_map<std::string, SDR> phone_cache_;
  mutable std::unordered_map<std::string, SDR> concept_cache_;
};

}  // namespace bm