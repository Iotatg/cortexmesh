#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/encoder.hpp"
#include "bm/sdr.hpp"

namespace bm {

struct Articulation {
  float jaw = 0.5f;
  float lip = 0.5f;
  float tongue = 0.5f;
  float tip = 0.5f;
  float velum = 0.0f;
  float voicing = 1.0f;
};

struct Formants {
  float f1 = 500.0f;
  float f2 = 1500.0f;
  float f3 = 2600.0f;
  float pitch = 120.0f;
};

class SpeechModel {
 public:
  explicit SpeechModel(size_t n = 1024, size_t k = 24, uint64_t seed = 31,
                      bool identity = true);

  SDR articulatory(const std::string& phone) const;
  SDR acoustic(const std::string& phone) const;
  SDR acoustic(const std::string& phone, const std::string& prev) const;

  Articulation articulation_of(const std::string& phone) const;
  Formants formants_of(const std::string& phone) const;

  void train(int epochs = 80);

  SDR predict_sound(const SDR& art) const;
  SDR predict_art(const SDR& sound) const;

  const std::vector<std::string>& phones() const { return phones_; }
  size_t size() const { return n_; }

 private:
  SDR block(size_t slice, size_t slices, float value, float lo, float hi, size_t width) const;

  size_t n_, k_;
  bool identity_;
  mutable Encoder enc_;
  std::vector<std::string> phones_;
  std::vector<Articulation> arts_;
  std::vector<Formants> forms_;
  mutable std::vector<SDR> art_cache_;
  mutable std::vector<SDR> snd_cache_;
  mutable std::unordered_map<std::string, SDR> ctx_cache_;
  AssociativeMemory fwd_;
  AssociativeMemory inv_;
};

}  // namespace bm