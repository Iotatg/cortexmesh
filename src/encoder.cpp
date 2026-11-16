#include "bm/encoder.hpp"

#include <algorithm>
#include <cmath>

namespace bm {

namespace {
constexpr size_t kBitsPerWord = 64;
}

Encoder::Encoder(size_t n, size_t k, uint64_t seed) : n_(n), k_(std::min(k, n)), seed_(seed) {}

SDR Encoder::hash_code(uint64_t h, size_t k, uint64_t salt) const {
  SDR s(n_, 0);
  Rng rng(h ^ (salt * 0x9E3779B97F4A7C15ull));
  size_t placed = 0;
  size_t guard = 0;
  while (placed < k && guard < k * 64) {
    ++guard;
    uint32_t idx = static_cast<uint32_t>((rng.next() >> 32) % n_);
    if (!s.test(idx)) {
      s.set(idx);
      ++placed;
    }
  }
  return s;
}

SDR Encoder::symbol(const std::string& sym) {
  auto it = cache_.find(sym);
  if (it != cache_.end()) return it->second;
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : sym) {
    h ^= c;
    h *= 1099511628211ull;
  }
  SDR code = hash_code(h, k_, seed_);
  cache_.emplace(sym, code);
  return code;
}


ScalarEncoder::ScalarEncoder(size_t n, size_t k, double min_v, double max_v, size_t buckets)
    : n_(n),
      k_(std::min(k, n)),
      buckets_(std::max<size_t>(buckets, 2)),
      min_v_(min_v),
      max_v_(max_v > min_v ? max_v : min_v + 1.0) {}

SDR ScalarEncoder::encode(double v) const {
  double t = (v - min_v_) / (max_v_ - min_v_);
  t = std::min(1.0, std::max(0.0, t));
  double pos = t * static_cast<double>(buckets_ - 1);
  size_t centre = static_cast<size_t>(pos + 0.5);
  size_t radius = std::max<size_t>(1, n_ / (buckets_ * 6));
  SDR s(n_, 0);
  for (size_t d = 0; d < n_; ++d) {
    size_t diff = (d > centre) ? (d - centre) : (centre - d);
    if (diff <= radius) s.set(static_cast<uint32_t>(d));
  }
  return s;
}

SDR Encoder::grid(double x, double y, size_t modules, size_t bits_per_module) {
  SDR s(n_, 0);
  for (size_t m = 0; m < modules && (m + 1) * bits_per_module <= n_; ++m) {
    double scale = std::pow(2.0, static_cast<double>(m)) * 4.0;
    double phase_x = x / scale;
    double phase_y = y / scale;
    auto fold = [](double p) {
      p = p - std::floor(p);
      return static_cast<size_t>(p * 1000.0) % 1000;
    };
    uint64_t h = 1469598103934665603ull;
    h ^= static_cast<uint64_t>(fold(phase_x));
    h *= 1099511628211ull;
    h ^= static_cast<uint64_t>(fold(phase_y) + 7919ull * m);
    h *= 1099511628211ull;
    Rng rng(h);
    for (size_t b = 0; b < bits_per_module; ++b) {
      uint32_t idx = static_cast<uint32_t>(m * bits_per_module + rng.below(static_cast<uint32_t>(bits_per_module)));
      s.set(idx);
    }
  }
  return s;
}

SDR Encoder::context(const std::vector<std::string>& history, size_t order) {
  std::vector<std::string> window;
  size_t start = history.size() > order ? history.size() - order : 0;
  for (size_t i = start; i < history.size(); ++i) window.push_back(history[i]);
  SDR acc(n_, 0);
  for (size_t i = 0; i < window.size(); ++i) {
    SDR part = symbol(window[i]);
    if (i + 1 < window.size()) {
      SDR shifted(n_, 0);
      for (uint32_t idx : part.indices()) {
        shifted.set(static_cast<uint32_t>((idx + 31 * (i + 1)) % n_));
      }
      part = shifted;
    }
    acc = acc | part;
  }
  return acc;
}

SDR Encoder::conjunction(const std::vector<SDR>& parts) const {
  SDR out(n_, 0);
  if (parts.empty()) return out;
  uint32_t salt = 0x9E3779B9u;
  for (size_t pi = 0; pi < parts.size(); ++pi) {
    for (uint32_t idx : parts[pi].indices()) {
      uint32_t j = static_cast<uint32_t>((static_cast<uint64_t>(idx) + static_cast<uint64_t>(salt) * (pi + 1)) % n_);
      out.set(j);
    }
    salt = salt * 1664525u + 1013904223u;
  }
  size_t c = out.count();
  if (c <= k_) return out;
  SDR trimmed(n_, 0);
  size_t stride = (c + k_ - 1) / k_;
  auto idxs = out.indices();
  for (size_t i = 0; i < idxs.size(); i += stride) trimmed.set(idxs[i]);
  return trimmed;
}

SDR Encoder::bundle(const std::vector<SDR>& parts) const {
  SDR out(n_, 0);
  if (parts.empty()) return out;
  std::vector<uint16_t> votes(n_, 0);
  for (const SDR& p : parts) {
    for (uint32_t i : p.indices()) {
      if (votes[i] < 65535) ++votes[i];
    }
  }
  std::vector<uint32_t> order(n_);
  for (size_t i = 0; i < n_; ++i) order[i] = static_cast<uint32_t>(i);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    if (votes[a] != votes[b]) return votes[a] > votes[b];
    return a < b;
  });
  for (size_t i = 0; i < k_ && i < order.size(); ++i) {
    if (votes[order[i]] == 0) break;
    out.set(order[i]);
  }
  return out;
}

}  // namespace bm