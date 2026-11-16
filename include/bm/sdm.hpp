#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bm/sdr.hpp"

namespace bm {

// Kanerva sparse distributed memory with hard locations.
//
// A is the number of hard locations. Each location owns a C-bit hard-location address
// (fixed at construction from `seed`, C/2 ones) and a C-bit data register. Both the
// address and the data are indexed by the same C-bit space as the input SDR, so the
// address overlap is a Hamming similarity in the same space as the data.
//
// Write is the Kanerva rule: for every location whose address overlaps the input SDR by
// at least `threshold` bits, XOR the data into that location's register. XOR is used
// instead of additive counters because it is 1 bit per cell (the RAM formula below), is
// local, and needs no learning rate.
//
// Read is a majority vote over the same selected registers. The literal rule "more than
// half of the selected registers hold the bit" works on a clean cue, because the cue wrote
// every selected location, so its own bit is present in all of them (measured: 40/40 bits
// returned at 100 patterns). It fails as soon as the cue is noisy: the read set is then
// only partly the write set, the bit survives in about half of the registers, and the vote
// sits at chance (measured 0.525 recall at 10 % noise, 0.000 at 25 %).
//
// The default read therefore compares the ones count with what chance predicts for a
// register written m times by words of density rho: q(m) = (1 - (1 - 2 rho)^m) / 2. A bit
// is on when its ones count exceeds
//     sum_a q(m_a) + z * sqrt(sum_a q(m_a) (1 - q(m_a))).
// For rho = 0.5 and m = 1 this is exactly "half of the selected locations"; for sparse
// words it is that rule with the sparsity and the load taken out. SdmRead::HalfSelected
// keeps the literal rule available as a control.
//
// RAM: A * C / 8 bytes of registers + A * C / 8 bytes of address table + A write counts.
struct SdmParams {
  size_t addresses = 4096;   // A: hard locations
  size_t width = 1024;       // C: address width == data width == SDR width
  size_t threshold = 28;     // minimum address overlap (bits) to select a location
  uint32_t seed = 0x5D1E5A17u;
};

enum class SdmRead {
  Majority,      // load- and sparsity-normalised majority (default)
  HalfSelected,  // literal Kanerva rule: ones > selected / 2 (control)
};

class Sdm {
 public:
  explicit Sdm(const SdmParams& params = SdmParams{});

  void write(const SDR& address, const SDR& data);
  SDR read(const SDR& address) const;
  SDR read(const SDR& address, SdmRead rule) const;
  size_t selected(const SDR& address) const;

  size_t addresses() const { return A_; }
  size_t width() const { return C_; }
  size_t threshold() const { return T_; }

  size_t address_bytes() const { return addr_.size() * sizeof(uint64_t); }
  size_t register_bytes() const { return reg_.size() * sizeof(uint64_t); }
  size_t count_bytes() const { return writes_.size(); }
  size_t ram_bytes() const { return address_bytes() + register_bytes() + count_bytes(); }

  static constexpr double kReadZ = 3.0;

 private:
  std::vector<uint32_t> select(const SDR& address) const;

  size_t A_ = 0, C_ = 0, T_ = 0, words_ = 0;
  std::vector<uint64_t> addr_;
  std::vector<uint64_t> reg_;
  std::vector<uint8_t> writes_;
};

}  // namespace bm