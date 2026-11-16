#include "bm/speech.hpp"

#include <algorithm>
#include <cmath>

namespace bm {

namespace {

struct PhoneSpec {
  const char* name;
  Articulation art;
  Formants form;
};

const std::vector<PhoneSpec>& Table() {
  static const std::vector<PhoneSpec> t = {
      {"p", {0.00f, 1.0f, 0.50f, 0.30f, 0.0f, 0.0f}, {300, 900, 2400, 120}},
      {"b", {0.00f, 1.0f, 0.50f, 0.30f, 0.0f, 1.0f}, {320, 950, 2400, 130}},
      {"m", {0.00f, 1.0f, 0.50f, 0.30f, 1.0f, 1.0f}, {280, 1000, 2300, 125}},
      {"t", {0.00f, 0.0f, 0.50f, 1.0f, 0.0f, 0.0f}, {350, 1700, 2600, 120}},
      {"d", {0.00f, 0.0f, 0.50f, 1.0f, 0.0f, 1.0f}, {370, 1750, 2600, 130}},
      {"n", {0.05f, 0.0f, 0.50f, 1.0f, 1.0f, 1.0f}, {300, 1700, 2500, 125}},
      {"k", {0.10f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, {400, 1200, 2500, 120}},
      {"g", {0.10f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}, {420, 1250, 2500, 130}},
      {"s", {0.10f, 0.0f, 0.60f, 1.0f, 0.0f, 0.0f}, {380, 1800, 2700, 120}},
      {"sh", {0.15f, 0.8f, 0.9f, 0.5f, 0.0f, 0.0f}, {400, 1900, 2600, 120}},
      {"h", {0.20f, 0.0f, 0.50f, 0.4f, 0.0f, 0.0f}, {450, 1500, 2500, 115}},
      {"l", {0.15f, 0.0f, 0.60f, 1.0f, 0.0f, 1.0f}, {380, 1600, 2600, 130}},
      {"r", {0.20f, 0.2f, 0.7f, 0.8f, 0.0f, 1.0f}, {400, 1400, 2500, 130}},
      {"y", {0.20f, 1.0f, 0.9f, 0.3f, 0.0f, 1.0f}, {350, 2000, 2600, 135}},
      {"v", {0.10f, 1.0f, 0.6f, 0.5f, 0.0f, 1.0f}, {380, 1500, 2500, 130}},
      {"aa", {1.0f, 0.0f, 0.5f, 0.4f, 0.0f, 1.0f}, {800, 1100, 2500, 120}},
      {"a", {0.7f, 0.0f, 0.5f, 0.4f, 0.0f, 1.0f}, {700, 1300, 2500, 120}},
      {"ii", {0.2f, 0.0f, 0.0f, 0.3f, 0.0f, 1.0f}, {300, 2300, 2900, 125}},
      {"u", {0.2f, 1.0f, 1.0f, 0.3f, 0.0f, 1.0f}, {320, 800, 2400, 120}},
      {"ee", {0.3f, 0.0f, 0.1f, 0.3f, 0.0f, 1.0f}, {400, 2100, 2700, 125}},
      {"o", {0.6f, 1.0f, 0.9f, 0.4f, 0.0f, 1.0f}, {550, 900, 2500, 120}},
      {"ai", {0.8f, 0.0f, 0.1f, 0.4f, 0.0f, 1.0f}, {750, 2000, 2700, 120}},
      {"au", {0.8f, 1.0f, 1.0f, 0.4f, 0.0f, 1.0f}, {700, 850, 2500, 120}},
      {"bh", {0.00f, 1.0f, 0.50f, 0.30f, 0.0f, 1.0f}, {340, 1000, 2400, 130}},
      {"th", {0.00f, 0.0f, 0.50f, 1.0f, 0.0f, 0.0f}, {360, 1750, 2600, 120}},
      {"kh", {0.10f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, {420, 1250, 2500, 120}},
      {"gh", {0.10f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}, {440, 1300, 2500, 130}},
      {"ch", {0.15f, 0.8f, 0.9f, 0.5f, 0.0f, 0.0f}, {400, 1900, 2600, 120}},
      {"jh", {0.15f, 0.8f, 0.9f, 0.5f, 0.0f, 1.0f}, {420, 1950, 2600, 130}},
      {"ph", {0.00f, 1.0f, 0.50f, 0.30f, 0.0f, 0.0f}, {310, 920, 2400, 120}},
      {"oo", {0.20f, 1.0f, 1.0f, 0.30f, 0.0f, 1.0f}, {320, 800, 2400, 120}},
      {"uu", {0.20f, 1.0f, 1.0f, 0.30f, 0.0f, 1.0f}, {320, 800, 2400, 120}},
      {"e", {0.30f, 0.0f, 0.1f, 0.30f, 0.0f, 1.0f}, {400, 2100, 2700, 125}},
      {"z", {0.10f, 0.0f, 0.6f, 1.0f, 0.0f, 1.0f}, {390, 1850, 2700, 130}},
  };
  return t;
}

float Place(const std::string& p) {
  if (p == "p" || p == "b" || p == "m") return 0.0f;
  if (p == "t" || p == "d" || p == "n" || p == "l" || p == "s") return 3.0f;
  if (p == "k" || p == "g") return 6.0f;
  if (p == "sh" || p == "y" || p == "v") return 5.0f;
  return 4.0f;
}

SDR FirstK(const SDR& s, size_t k) {
  SDR o(s.size(), 0);
  auto idx = s.indices();
  for (size_t i = 0; i < k && i < idx.size(); ++i) o.set(idx[i]);
  return o;
}

}  // namespace

SpeechModel::SpeechModel(size_t n, size_t k, uint64_t seed, bool identity)
    : n_(n),
      k_(std::min(k, n)),
      identity_(identity),
      enc_(n, k, seed),
      fwd_(n, n, k, 0.08f),
      inv_(n, n, k, 0.08f) {
  for (const auto& p : Table()) {
    phones_.push_back(p.name);
    arts_.push_back(p.art);
    forms_.push_back(p.form);
    art_cache_.emplace_back(n_, 0);
    snd_cache_.emplace_back(n_, 0);
  }
  for (size_t i = 0; i < phones_.size(); ++i) {
    art_cache_[i] = articulatory(phones_[i]);
    snd_cache_[i] = acoustic(phones_[i]);
  }
}

SDR SpeechModel::block(size_t slice, size_t slices, float value, float lo, float hi,
                       size_t width) const {
  size_t slice_bits = n_ / slices;
  size_t base = slice * slice_bits;
  size_t span = slice_bits > width ? slice_bits - width : 1;
  float t = hi > lo ? (value - lo) / (hi - lo) : 0.0f;
  t = std::min(1.0f, std::max(0.0f, t));
  size_t pos = static_cast<size_t>(t * static_cast<float>(span));
  SDR s(n_, 0);
  for (size_t i = 0; i < width && base + pos + i < n_; ++i) {
    s.set(static_cast<uint32_t>(base + pos + i));
  }
  return s;
}

SDR SpeechModel::articulatory(const std::string& phone) const {
  for (size_t i = 0; i < phones_.size(); ++i) {
    if (phones_[i] == phone) {
      const Articulation& a = arts_[i];
      std::vector<SDR> parts = {
          block(0, 6, a.jaw, 0, 1, 2),   block(1, 6, a.lip, 0, 1, 2),
          block(2, 6, a.tongue, 0, 1, 2), block(3, 6, a.tip, 0, 1, 2),
          block(4, 6, a.velum, 0, 1, 2), block(5, 6, a.voicing, 0, 1, 2),
          identity_ ? FirstK(enc_.symbol("art:" + phone), 12) : SDR(n_, 0),
      };
      SDR s(n_, 0);
      for (const SDR& p : parts) s |= p;
      return s;
    }
  }
  return enc_.symbol("ph?" + phone);
}

SDR SpeechModel::acoustic(const std::string& phone) const {
  for (size_t i = 0; i < phones_.size(); ++i) {
    if (phones_[i] == phone) {
      const Formants& f = forms_[i];
      std::vector<SDR> parts = {
          block(0, 4, f.f1, 200, 1000, 2),  block(1, 4, f.f2, 700, 2400, 2),
          block(2, 4, f.f3, 2000, 3200, 2), block(3, 4, f.pitch, 80, 200, 2),
          identity_ ? FirstK(enc_.symbol("snd:" + phone), 12) : SDR(n_, 0),
      };
      SDR s(n_, 0);
      for (const SDR& p : parts) s |= p;
      return s;
    }
  }
  return enc_.symbol("snd?" + phone);
}

SDR SpeechModel::acoustic(const std::string& phone, const std::string& prev) const {
  std::string key = prev + "|" + phone;
  auto it = ctx_cache_.find(key);
  if (it != ctx_cache_.end()) return it->second;
  SDR base = acoustic(phone);
  // co-articulation: the previous place pulls F2 toward it
  float shift = (Place(prev) - 3.0f) * 60.0f;
  float f2 = formants_of(phone).f2 + shift;
  f2 = std::min(2400.0f, std::max(700.0f, f2));
  const Formants f = formants_of(phone);
  std::vector<SDR> parts = {
      block(0, 4, f.f1, 200, 1000, 2),
      block(1, 4, f2, 700, 2400, 2),
      block(2, 4, f.f3, 2000, 3200, 2),
      block(3, 4, f.pitch, 80, 200, 2),
      identity_ ? FirstK(enc_.symbol("snd:" + phone), 12) : SDR(n_, 0),
  };
  SDR s(n_, 0);
  for (const SDR& p : parts) s |= p;
  (void)base;
  ctx_cache_.emplace(key, s);
  return s;
}

Articulation SpeechModel::articulation_of(const std::string& phone) const {
  for (size_t i = 0; i < phones_.size(); ++i) {
    if (phones_[i] == phone) return arts_[i];
  }
  return Articulation{};
}

Formants SpeechModel::formants_of(const std::string& phone) const {
  for (size_t i = 0; i < phones_.size(); ++i) {
    if (phones_[i] == phone) return forms_[i];
  }
  return Formants{};
}

void SpeechModel::train(int epochs) {
  for (int e = 0; e < epochs; ++e) {
    for (size_t i = 0; i < phones_.size(); ++i) {
      fwd_.learn(art_cache_[i], snd_cache_[i]);
      inv_.learn(snd_cache_[i], art_cache_[i]);
    }
  }
}

SDR SpeechModel::predict_sound(const SDR& art) const { return fwd_.recall(art); }

SDR SpeechModel::predict_art(const SDR& sound) const { return inv_.recall(sound); }

}  // namespace bm