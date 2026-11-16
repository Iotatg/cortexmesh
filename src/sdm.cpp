#include "bm/sdm.hpp"

#include <algorithm>
#include <cmath>

namespace bm {

namespace {

constexpr size_t kBitsPerWord = 64;

inline size_t WordsFor(size_t bits) { return (bits + kBitsPerWord - 1) / kBitsPerWord; }

// Software popcount. The build has no -mpopcnt, so __builtin_popcountll would be a call
// into libgcc (about 10x slower); this is inlined and stays portable.
inline size_t Popcnt(uint64_t x) {
  x = x - ((x >> 1) & 0x5555555555555555ull);
  x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
  x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
  return static_cast<size_t>((x * 0x0101010101010101ull) >> 56);
}

// Expected fraction of ones in a register that was written by m independent words of
// density rho (XOR of m Bernoulli(rho) bits is Bernoulli(q)).
inline double QOf(double rho, double m) {
  return 0.5 * (1.0 - std::pow(1.0 - 2.0 * rho, m));
}

}  // namespace

Sdm::Sdm(const SdmParams& params)
    : A_(params.addresses ? params.addresses : 1),
      C_(params.width ? params.width : 1),
      T_(params.threshold),
      words_(WordsFor(C_)),
      addr_(A_ * words_, 0),
      reg_(A_ * words_, 0),
      writes_(A_, 0) {
  Rng rng(params.seed);
  const uint32_t half = static_cast<uint32_t>(C_ / 2);
  for (size_t a = 0; a < A_; ++a) {
    SDR address(C_, 0);
    address.fill_k(half, rng);
    std::copy(address.words().begin(), address.words().end(), addr_.begin() + a * words_);
  }
}

std::vector<uint32_t> Sdm::select(const SDR& address) const {
  std::vector<uint32_t> out;
  if (address.size() != C_) return out;
  const uint64_t* aw = address.words().data();
  for (size_t a = 0; a < A_; ++a) {
    const uint64_t* ad = addr_.data() + a * words_;
    size_t o0 = 0, o1 = 0, o2 = 0, o3 = 0;
    size_t w = 0;
    for (; w + 4 <= words_; w += 4) {
      o0 += Popcnt(ad[w] & aw[w]);
      o1 += Popcnt(ad[w + 1] & aw[w + 1]);
      o2 += Popcnt(ad[w + 2] & aw[w + 2]);
      o3 += Popcnt(ad[w + 3] & aw[w + 3]);
    }
    for (; w < words_; ++w) o0 += Popcnt(ad[w] & aw[w]);
    const size_t overlap = o0 + o1 + o2 + o3;
    if (overlap >= T_) out.push_back(static_cast<uint32_t>(a));
  }
  return out;
}

size_t Sdm::selected(const SDR& address) const { return select(address).size(); }

void Sdm::write(const SDR& address, const SDR& data) {
  if (address.size() != C_ || data.size() != C_) return;
  std::vector<uint32_t> sel = select(address);
  const uint64_t* dw = data.words().data();
  for (uint32_t a : sel) {
    uint64_t* r = reg_.data() + static_cast<size_t>(a) * words_;
    for (size_t w = 0; w < words_; ++w) r[w] ^= dw[w];
    if (writes_[a] < 255) ++writes_[a];
  }
}

SDR Sdm::read(const SDR& address) const { return read(address, SdmRead::Majority); }

SDR Sdm::read(const SDR& address, SdmRead rule) const {
  SDR out(C_, 0);
  std::vector<uint32_t> sel = select(address);
  if (sel.empty()) return out;

  std::vector<uint32_t> ones(C_, 0);
  for (uint32_t a : sel) {
    const uint64_t* r = reg_.data() + static_cast<size_t>(a) * words_;
    for (size_t w = 0; w < words_; ++w) {
      uint64_t bits = r[w];
      while (bits) {
        const uint32_t b = static_cast<uint32_t>(__builtin_ctzll(bits));
        ++ones[w * kBitsPerWord + b];
        bits &= bits - 1;
      }
    }
  }

  if (rule == SdmRead::HalfSelected) {
    const uint32_t half = static_cast<uint32_t>(sel.size() / 2);
    for (size_t j = 0; j < C_; ++j) {
      if (ones[j] > half) out.set(static_cast<uint32_t>(j));
    }
    return out;
  }

  const double rho =
      address.count() ? static_cast<double>(address.count()) / static_cast<double>(C_) : 0.0;
  double mean = 0.0, var = 0.0;
  for (uint32_t a : sel) {
    const double q = QOf(rho, static_cast<double>(writes_[a]));
    mean += q;
    var += q * (1.0 - q);
  }
  const double thr = mean + kReadZ * std::sqrt(var);
  for (size_t j = 0; j < C_; ++j) {
    if (static_cast<double>(ones[j]) > thr) out.set(static_cast<uint32_t>(j));
  }
  return out;
}

}  // namespace bm