#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "bm/sdr.hpp"

namespace bm {

class Encoder {
 public:
  Encoder(size_t n, size_t k, uint64_t seed = 1);

  SDR symbol(const std::string& s);
  SDR scalar(double value, double min_v, double max_v, size_t buckets);
  SDR grid(double x, double y, size_t modules, size_t bits_per_module);
  SDR context(const std::vector<std::string>& history, size_t order);
  SDR conjunction(const std::vector<SDR>& parts) const;
  SDR bundle(const std::vector<SDR>& parts) const;

  size_t size() const { return n_; }
  size_t k() const { return k_; }

 private:
  SDR hash_code(uint64_t h, size_t k, uint64_t salt) const;
  size_t n_, k_;
  uint64_t seed_;
  std::unordered_map<std::string, SDR> cache_;
};

class ScalarEncoder {
 public:
  ScalarEncoder(size_t n, size_t k, double min_v, double max_v, size_t buckets);

  SDR encode(double v) const;
  size_t size() const { return n_; }

 private:
  size_t n_, k_, buckets_;
  double min_v_, max_v_;
};

}  // namespace bm