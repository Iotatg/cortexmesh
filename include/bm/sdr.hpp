#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bm {

class Rng {
 public:
  explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) : s_(seed ? seed : 1) {}
  uint64_t next() {
    uint64_t z = (s_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint32_t below(uint32_t n) { return n ? static_cast<uint32_t>(next() % n) : 0; }
  double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

 private:
  uint64_t s_;
};

class SDR {
 public:
  SDR() = default;
  explicit SDR(size_t n, size_t k = 0);

  static SDR random(size_t n, size_t k, uint64_t seed);
  static SDR from_indices(size_t n, const std::vector<uint32_t>& idx);
  static SDR from_bits(size_t n, const std::vector<uint64_t>& words);

  size_t size() const { return n_; }
  size_t count() const;
  bool empty() const { return count() == 0; }

  void set(uint32_t i);
  void reset(uint32_t i);
  bool test(uint32_t i) const;
  void clear();
  void fill_k(uint32_t k, Rng& rng);

  SDR operator&(const SDR& o) const;
  SDR operator|(const SDR& o) const;
  SDR& operator|=(const SDR& o);
  SDR operator^(const SDR& o) const;
  SDR operator~() const;

  size_t overlap(const SDR& o) const;
  size_t intersection() const { return count(); }
  double jaccard(const SDR& o) const;
  double similarity(const SDR& o) const;

  bool operator==(const SDR& o) const;

  std::vector<uint32_t> indices() const;
  const std::vector<uint64_t>& words() const { return w_; }

  std::string to_string() const;
  uint64_t hash() const;

 private:
  size_t n_ = 0;
  std::vector<uint64_t> w_;
};

size_t popcount_words(const std::vector<uint64_t>& w);
bool is_sparse_valid(const SDR& s, double min_sparsity, double max_sparsity);

}  // namespace bm