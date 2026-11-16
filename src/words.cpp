#include "bm/words.hpp"

#include <algorithm>
#include <cmath>

namespace bm {

namespace {

struct PhoneFeatures {
  uint8_t place;    // 0 bilabial .. 7 glottal
  uint8_t manner;   // 0 stop 1 nasal 2 fricative 3 approximant 4 vowel
  uint8_t voiced;   // 0 voiceless 1 voiced
  uint8_t height;   // 0 close .. 5 central (vowels only)
};

const std::unordered_map<std::string, PhoneFeatures>& PhoneTable() {
  static const std::unordered_map<std::string, PhoneFeatures> t = {
      {"p", {0, 0, 0, 5}},   {"ph", {0, 0, 0, 5}},  {"b", {0, 0, 1, 5}},
      {"bh", {0, 0, 1, 5}},  {"t", {2, 0, 0, 5}},   {"th", {2, 0, 0, 5}},
      {"d", {2, 0, 1, 5}},   {"dh", {2, 0, 1, 5}},  {"k", {6, 0, 0, 5}},
      {"kh", {6, 0, 0, 5}},  {"g", {6, 0, 1, 5}},   {"gh", {6, 0, 1, 5}},
      {"ch", {5, 0, 0, 5}},  {"j", {5, 0, 1, 5}},   {"m", {0, 1, 1, 5}},
      {"n", {3, 1, 1, 5}},   {"l", {3, 3, 1, 5}},   {"r", {3, 3, 1, 5}},
      {"s", {3, 2, 0, 5}},   {"sh", {5, 2, 0, 5}},  {"h", {7, 2, 0, 5}},
      {"y", {5, 3, 1, 5}},   {"v", {1, 2, 1, 5}},   {"z", {3, 2, 1, 5}},
      {"ng", {6, 1, 1, 5}},  {"aa", {5, 4, 1, 4}},  {"a", {5, 4, 1, 3}},
      {"ii", {0, 4, 1, 0}},  {"u", {7, 4, 1, 0}},   {"uu", {7, 4, 1, 1}},
      {"ee", {0, 4, 1, 2}},  {"e", {0, 4, 1, 3}},   {"o", {7, 4, 1, 2}},
      {"ai", {0, 4, 1, 4}},  {"au", {7, 4, 1, 4}},
      {"oo", {7, 4, 1, 1}},  {"jh", {5, 0, 1, 5}},
      {"<s>", {5, 5, 0, 5}}, {"</s>", {5, 5, 0, 5}},
  };
  return t;
}

const std::unordered_map<std::string, std::vector<std::string>>& WordTable() {
  static const std::unordered_map<std::string, std::vector<std::string>> t = {
      {"main", {"m", "aa", "n"}},
      {"bhookha", {"bh", "uu", "kh", "aa"}},
      {"bhookh", {"bh", "uu", "kh"}},
      {"thak", {"th", "a"}},
      {"thakan", {"th", "a", "k", "a", "n"}},
      {"gaya", {"g", "aa", "y", "aa"}},
      {"khush", {"kh", "u", "sh"}},
      {"theek", {"th", "ee", "k"}},
      {"paani", {"p", "aa", "n", "ii"}},
      {"chahiye", {"ch", "a", "h", "ii", "y", "e"}},
      {"dard", {"d", "a", "r", "d"}},
      {"raha", {"r", "a", "h", "a"}},
      {"hai", {"h", "ai"}},
      {"neend", {"n", "ee", "n", "d"}},
      {"aa", {"aa"}},
      {"hoon", {"h", "oo", "n"}},
      {"mujhe", {"m", "u", "jh", "e"}},
      {"rahi", {"r", "a", "h", "ii"}},
      {"lagi", {"l", "a", "g", "ii"}},
      {"tum", {"t", "u", "m"}},
      {"kaise", {"k", "ai", "s", "e"}},
      {"ho", {"h", "o"}},
      {"shukriya", {"sh", "u", "k", "r", "ii", "y", "a"}},
      {"madad", {"m", "a", "d", "a", "d"}},
      {"pyaas", {"p", "y", "aa", "s"}},
      {"<s>", {"<s>"}},
      {"</s>", {"</s>"}},
  };
  return t;
}

const std::unordered_map<std::string, std::vector<std::string>>& ConceptTable() {
  static const std::unordered_map<std::string, std::vector<std::string>> t = {
      {"main", {"self"}},
      {"tum", {"other"}},
      {"ho", {"self"}},
      {"hai", {"self"}},
      {"bhookha", {"body", "need", "negative"}},
      {"bhookh", {"body", "need", "negative"}},
      {"thak", {"body", "negative"}},
      {"thakan", {"body", "need", "negative"}},
      {"khush", {"positive"}},
      {"theek", {"body", "positive"}},
      {"paani", {"liquid", "need"}},
      {"pyaas", {"liquid", "need", "negative"}},
      {"chahiye", {"need"}},
      {"dard", {"body", "negative"}},
      {"raha", {"action"}},
      {"rahi", {"action"}},
      {"aa", {"action"}},
      {"hoon", {"self"}},
      {"mujhe", {"self"}},
      {"lagi", {"need"}},
      {"neend", {"body", "need", "negative"}},
      {"kaise", {"other"}},
      {"shukriya", {"positive", "other"}},
      {"madad", {"need", "other"}},
  };
  return t;
}

float Valence(const std::string& w) {
  if (w == "bhookh" || w == "bhookha") return -0.6f;
  if (w == "dard" || w == "thak" || w == "thakan") return -0.5f;
  if (w == "neend") return -0.2f;
  if (w == "khush" || w == "shukriya" || w == "theek") return 0.8f;
  if (w == "paani" || w == "pyaas") return 0.3f;
  if (w == "madad") return -0.2f;
  return 0.0f;
}

float Arousal(const std::string& w) {
  if (w == "dard") return 0.8f;
  if (w == "bhookh" || w == "bhookha") return 0.6f;
  if (w == "khush") return 0.6f;
  if (w == "thak" || w == "thakan" || w == "neend") return 0.1f;
  if (w == "madad" || w == "chahiye") return 0.5f;
  return 0.25f;
}



SDR UnionTrim(const std::vector<SDR>& parts, size_t n, size_t k) {
  SDR u(n, 0);
  for (const SDR& p : parts) u |= p;
  auto idx = u.indices();
  SDR out(n, 0);
  if (idx.empty()) return out;
  size_t take = std::min(k, idx.size());
  size_t stride = std::max<size_t>(1, idx.size() / take);
  for (size_t i = 0; i < idx.size() && out.count() < take; i += stride) out.set(idx[i]);
  return out;
}

}  // namespace

WordEncoder::WordEncoder(size_t n, size_t k, uint64_t seed)
    : n_(n), k_(std::min(k, n)), seed_(seed), enc_(n, k, seed) {
  for (const auto& [w, ph] : WordTable()) {
    WordEntry e;
    e.phones = ph;
    auto it = ConceptTable().find(w);
    if (it != ConceptTable().end()) e.concepts = it->second;
    e.valence = Valence(w);
    e.arousal = Arousal(w);
    words_.emplace(w, std::move(e));
    vocab_.push_back(w);
  }
  std::sort(vocab_.begin(), vocab_.end());
}

SDR WordEncoder::feature_code(size_t slice_index, size_t slice_count, size_t value,
                              size_t max_value, size_t width) const {
  size_t slice = n_ / slice_count;
  size_t base = slice_index * slice;
  size_t span = slice > width ? slice - width : 1;
  size_t pos = max_value > 0 ? (value * span) / max_value : 0;
  SDR s(n_, 0);
  for (size_t i = 0; i < width && base + pos + i < n_; ++i) s.set(static_cast<uint32_t>(base + pos + i));
  return s;
}

SDR WordEncoder::phone(const std::string& p) const {
  auto it = phone_cache_.find(p);
  if (it != phone_cache_.end()) return it->second;
  auto ft = PhoneTable().find(p);
  if (ft == PhoneTable().end()) {
    SDR s = enc_.symbol("ph?" + p);
    phone_cache_.emplace(p, s);
    return s;
  }
  const PhoneFeatures& f = ft->second;
  SDR uniq = enc_.symbol("ph:" + p);
  std::vector<SDR> parts = {
      uniq,
      feature_code(0, 4, f.place, 7, 1),
      feature_code(1, 4, f.manner, 4, 1),
      feature_code(2, 4, f.voiced, 1, 1),
      feature_code(3, 4, f.height, 5, 1),
  };
  SDR s = UnionTrim(parts, n_, 20);
  phone_cache_.emplace(p, s);
  return s;
}

SDR WordEncoder::concept_code(const std::string& c) const {
  auto it = concept_cache_.find(c);
  if (it != concept_cache_.end()) return it->second;
  static const std::vector<std::string> dims = {"self",  "other",   "body", "need",
                                                "pos",   "neg",     "liquid", "action"};
  int idx = -1;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == c || (c == "positive" && dims[i] == "pos") ||
        (c == "negative" && dims[i] == "neg")) {
      idx = static_cast<int>(i);
      break;
    }
  }
  if (idx < 0) {
    SDR s = enc_.symbol("cx?" + c);
    concept_cache_.emplace(c, s);
    return s;
  }
  SDR s = feature_code(static_cast<size_t>(idx), 8, 1, 1, 4);
  concept_cache_.emplace(c, s);
  return s;
}

SDR WordEncoder::valence_code(float v) const {
  int val = static_cast<int>(std::lround((v + 1.0f) * 5.0f));
  val = std::max(0, std::min(10, val));
  return feature_code(4, 8, static_cast<size_t>(val), 10, 12);
}

SDR WordEncoder::arousal_code(float v) const {
  int val = static_cast<int>(std::lround(v * 10.0f));
  val = std::max(0, std::min(10, val));
  return feature_code(5, 8, static_cast<size_t>(val), 10, 12);
}

SDR WordEncoder::encode(const std::string& word) const {
  auto it = words_.find(word);
  if (it == words_.end()) return enc_.symbol("word?" + word);
  const WordEntry& e = it->second;
  std::vector<SDR> parts;
  parts.reserve(e.phones.size() + e.concepts.size() + 2);
  for (const auto& p : e.phones) parts.push_back(phone(p));
  for (const auto& c : e.concepts) parts.push_back(concept_code(c));
  parts.push_back(valence_code(e.valence));
  parts.push_back(arousal_code(e.arousal));
  return UnionTrim(parts, n_, k_);
}

const WordEntry* WordEncoder::entry(const std::string& word) const {
  auto it = words_.find(word);
  return it == words_.end() ? nullptr : &it->second;
}

}  // namespace bm