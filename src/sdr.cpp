#include "bm/sdr.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace bm {

namespace {
constexpr size_t kBitsPerWord = 64;

inline size_t words_for(size_t n) { return (n + kBitsPerWord - 1) / kBitsPerWord; }
}  // namespace

SDR::SDR(size_t n, size_t k) : n_(n), w_(words_for(n), 0) {
  if (k > n) k = n;
  if (k > 0 && n > 0) {
    Rng rng(n * 2654435761u + k);
    fill_k(static_cast<uint32_t>(k), rng);
  }
}

SDR SDR::random(size_t n, size_t k, uint64_t seed) {
  SDR s(n, 0);
  Rng rng(seed);
  s.fill_k(static_cast<uint32_t>(k), rng);
  return s;
}

SDR SDR::from_indices(size_t n, const std::vector<uint32_t>& idx) {
  SDR s(n, 0);
  for (uint32_t i : idx) s.set(i);
  return s;
}

SDR SDR::from_bits(size_t n, const std::vector<uint64_t>& words) {
  SDR s(n, 0);
  size_t m = std::min(words.size(), s.w_.size());
  for (size_t i = 0; i < m; ++i) s.w_[i] = words[i];
  return s;
}

size_t SDR::count() const { return popcount_words(w_); }

void SDR::set(uint32_t i) {
  if (i >= n_) return;
  w_[i / kBitsPerWord] |= (1ull << (i % kBitsPerWord));
}

void SDR::reset(uint32_t i) {
  if (i >= n_) return;
  w_[i / kBitsPerWord] &= ~(1ull << (i % kBitsPerWord));
}

bool SDR::test(uint32_t i) const {
  if (i >= n_) return false;
  return (w_[i / kBitsPerWord] >> (i % kBitsPerWord)) & 1ull;
}

void SDR::clear() { std::fill(w_.begin(), w_.end(), 0ull); }

void SDR::fill_k(uint32_t k, Rng& rng) {
  clear();
  if (n_ == 0 || k == 0) return;
  if (k >= n_) {
    for (size_t i = 0; i < n_; ++i) set(static_cast<uint32_t>(i));
    return;
  }
  size_t placed = 0;
  while (placed < k) {
    uint32_t idx = static_cast<uint32_t>((rng.next() >> 32) % n_);
    if (!test(idx)) {
      set(idx);
      ++placed;
    }
  }
}

SDR& SDR::operator|=(const SDR& o) {
  for (size_t i = 0; i < w_.size() && i < o.w_.size(); ++i) w_[i] |= o.w_[i];
  return *this;
}

SDR SDR::operator&(const SDR& o) const {
  SDR r(n_, 0);
  size_t m = std::min(w_.size(), o.w_.size());
  for (size_t i = 0; i < m; ++i) r.w_[i] = w_[i] & o.w_[i];
  return r;
}

SDR SDR::operator|(const SDR& o) const {
  SDR r(n_, 0);
  size_t m = std::min(w_.size(), o.w_.size());
  for (size_t i = 0; i < m; ++i) r.w_[i] = w_[i] | o.w_[i];
  return r;
}

SDR SDR::operator^(const SDR& o) const {
  SDR r(n_, 0);
  size_t m = std::min(w_.size(), o.w_.size());
  for (size_t i = 0; i < m; ++i) r.w_[i] = w_[i] ^ o.w_[i];
  return r;
}

SDR SDR::operator~() const {
  SDR r(n_, 0);
  for (size_t i = 0; i < w_.size(); ++i) {
    uint64_t tail = 0;
    size_t used = n_ - i * kBitsPerWord;
    if (used < kBitsPerWord && used > 0) {
      tail = (used == kBitsPerWord) ? 0 : ((1ull << used) - 1);
      r.w_[i] = (~w_[i]) & tail;
    } else {
      r.w_[i] = ~w_[i];
    }
  }
  return r;
}

size_t SDR::overlap(const SDR& o) const { return (*this & o).count(); }

double SDR::jaccard(const SDR& o) const {
  size_t inter = overlap(o);
  size_t uni = count() + o.count() - inter;
  return uni == 0 ? 0.0 : static_cast<double>(inter) / static_cast<double>(uni);
}

double SDR::similarity(const SDR& o) const {
  size_t c = count();
  if (c == 0) return 0.0;
  return static_cast<double>(overlap(o)) / static_cast<double>(c);
}

bool SDR::operator==(const SDR& o) const {
  if (n_ != o.n_) return false;
  return w_ == o.w_;
}

std::vector<uint32_t> SDR::indices() const {
  std::vector<uint32_t> out;
  out.reserve(count());
  for (size_t wi = 0; wi < w_.size(); ++wi) {
    uint64_t word = w_[wi];
    while (word) {
      uint32_t b = static_cast<uint32_t>(__builtin_ctzll(word));
      out.push_back(static_cast<uint32_t>(wi * kBitsPerWord + b));
      word &= word - 1;
    }
  }
  return out;
}

std::string SDR::to_string() const {
  std::ostringstream os;
  auto idx = indices();
  os << "{n=" << n_ << ", k=" << idx.size() << " [";
  size_t lim = std::min<size_t>(idx.size(), 24);
  for (size_t i = 0; i < lim; ++i) {
    if (i) os << ",";
    os << idx[i];
  }
  if (idx.size() > lim) os << ",...";
  os << "]}";
  return os.str();
}

uint64_t SDR::hash() const {
  uint64_t h = 1469598103934665603ull;
  for (uint64_t word : w_) {
    h ^= word;
    h *= 1099511628211ull;
  }
  h ^= n_;
  h *= 1099511628211ull;
  return h;
}

size_t popcount_words(const std::vector<uint64_t>& w) {
  size_t total = 0;
  for (uint64_t word : w) total += static_cast<size_t>(__builtin_popcountll(word));
  return total;
}

bool is_sparse_valid(const SDR& s, double min_sparsity, double max_sparsity) {
  if (s.size() == 0) return false;
  double frac = static_cast<double>(s.count()) / static_cast<double>(s.size());
  return frac >= min_sparsity && frac <= max_sparsity;
}

}  // namespace bm