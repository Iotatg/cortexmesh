#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "bm/sdr.hpp"
#include "bm/words.hpp"

using namespace bm;

namespace {

struct Score {
  int tp = 0, fp = 0, fn = 0;
  double precision() const { return tp + fp ? double(tp) / (tp + fp) : 0.0; }
  double recall() const { return tp + fn ? double(tp) / (tp + fn) : 0.0; }
  double f1() const {
    double p = precision(), r = recall();
    return p + r ? 2 * p * r / (p + r) : 0.0;
  }
};

bool IsVowel(const std::string& p) {
  static const char* v[] = {"aa", "a", "ii", "u", "ee", "o", "ai", "au", "oo", "uu", "e"};
  for (const char* x : v) {
    if (p == x) return true;
  }
  return false;
}

}  // namespace

int main() {
  WordEncoder we(512, 24, 11);

  // Corpus: word sequences, then a continuous phone stream with boundary labels.
  std::vector<std::vector<std::string>> sentences = {
      {"main", "bhookha", "hoon"},   {"main", "thak", "gaya", "hoon"},
      {"main", "khush", "hoon"},      {"main", "theek", "hoon"},
      {"mujhe", "paani", "chahiye"},  {"mujhe", "dard", "ho", "raha", "hai"},
      {"mujhe", "neend", "aa", "rahi", "hai"},
      {"mujhe", "bhookh", "lagi", "hai"},
      {"tum", "kaise", "ho"},         {"shukriya"},
      {"mujhe", "madad", "chahiye"},
  };

  std::vector<std::vector<std::string>> words;
  for (const auto& s : sentences) {
    for (const auto& w : s) {
      const WordEntry* e = we.entry(w);
      if (e) words.push_back(e->phones);
    }
  }

  // 1. transition statistics from within-word pairs only
  std::map<std::pair<std::string, std::string>, int> trans;
  std::map<std::string, int> uni;
  for (const auto& w : words) {
    for (size_t i = 0; i < w.size(); ++i) {
      uni[w[i]]++;
      if (i + 1 < w.size()) trans[{w[i], w[i + 1]}]++;
    }
  }
  const double alpha = 0.1;
  const double V = 40.0;
  auto p_next = [&](const std::string& a, const std::string& b) {
    auto it = trans.find({a, b});
    int c = it == trans.end() ? 0 : it->second;
    int ca = uni.count(a) ? uni[a] : 0;
    return (c + alpha) / (ca + alpha * V);
  };

  // stream and boundary labels
  std::vector<std::string> stream;
  std::vector<bool> boundary;  // boundary[i] = true if stream[i] starts a new word
  for (const auto& s : sentences) {
    for (size_t wi = 0; wi < s.size(); ++wi) {
      const WordEntry* e = we.entry(s[wi]);
      if (!e) continue;
      for (size_t pi = 0; pi < e->phones.size(); ++pi) {
        stream.push_back(e->phones[pi]);
        boundary.push_back(pi == 0);
      }
    }
    // sentence boundary: not a word boundary, skip
  }

  // sweep threshold on surprise = -log P(next|prev)
  double best_f1 = 0, best_thr = 0, best_p = 0, best_r = 0;
  for (double thr = 0.5; thr <= 6.0; thr += 0.1) {
    Score sc;
    for (size_t i = 1; i < stream.size(); ++i) {
      double surprise = -std::log(p_next(stream[i - 1], stream[i]) + 1e-12);
      bool pred = surprise > thr;
      if (pred && boundary[i]) ++sc.tp;
      if (pred && !boundary[i]) ++sc.fp;
      if (!pred && boundary[i]) ++sc.fn;
    }
    if (sc.f1() > best_f1) {
      best_f1 = sc.f1();
      best_thr = thr;
      best_p = sc.precision();
      best_r = sc.recall();
    }
  }
  printf("boundary,best_thr=%.2f,precision=%.3f,recall=%.3f,f1=%.3f\n", best_thr, best_p, best_r,
         best_f1);

  // vowel heuristic: boundary before every vowel
  {
    Score sc;
    for (size_t i = 1; i < stream.size(); ++i) {
      bool pred = IsVowel(stream[i]);
      if (pred && boundary[i]) ++sc.tp;
      if (pred && !boundary[i]) ++sc.fp;
      if (!pred && boundary[i]) ++sc.fn;
    }
    printf("baseline_vowel,precision=%.3f,recall=%.3f,f1=%.3f\n", sc.precision(), sc.recall(),
           sc.f1());
  }

  // 2. auditory categories: online k-means on noisy formant vectors
  {
    // collect formant vectors per phone (hand-set targets + noise)
    struct Sample {
      std::array<double, 4> x;
      int phone;
    };
    std::vector<std::string> phones;
    std::vector<std::array<double, 4>> targets;
    // reuse the speech model's formants through the word encoder's phone table? use a small table
    // (kept local to avoid depending on speech model internals)
    const std::vector<std::pair<std::string, std::array<double, 4>>> ft = {
        {"p", {300, 900, 2400, 120}},  {"b", {320, 950, 2400, 130}},
        {"m", {280, 1000, 2300, 125}}, {"t", {350, 1700, 2600, 120}},
        {"d", {370, 1750, 2600, 130}}, {"n", {300, 1700, 2500, 125}},
        {"k", {400, 1200, 2500, 120}}, {"g", {420, 1250, 2500, 130}},
        {"s", {380, 1800, 2700, 120}}, {"sh", {400, 1900, 2600, 120}},
        {"h", {450, 1500, 2500, 115}}, {"l", {380, 1600, 2600, 130}},
        {"r", {400, 1400, 2500, 130}}, {"y", {350, 2000, 2600, 135}},
        {"v", {380, 1500, 2500, 130}}, {"aa", {800, 1100, 2500, 120}},
        {"a", {700, 1300, 2500, 120}}, {"ii", {300, 2300, 2900, 125}},
        {"u", {320, 800, 2400, 120}},  {"ee", {400, 2100, 2700, 125}},
        {"o", {550, 900, 2500, 120}},  {"ai", {750, 2000, 2700, 120}},
        {"au", {700, 850, 2500, 120}},
    };
    for (size_t i = 0; i < ft.size(); ++i) {
      phones.push_back(ft[i].first);
      targets.push_back(ft[i].second);
    }

    Rng rng(99);
    std::vector<Sample> samples;
    for (size_t p = 0; p < phones.size(); ++p) {
      for (int rep = 0; rep < 20; ++rep) {
        Sample s;
        s.phone = static_cast<int>(p);
        for (int d = 0; d < 4; ++d) {
          double noise = (rng.unit() + rng.unit() + rng.unit() - 1.5) * 40.0;
          s.x[static_cast<size_t>(d)] = targets[p][static_cast<size_t>(d)] + noise;
        }
        samples.push_back(s);
      }
    }

    const int K = static_cast<int>(phones.size());
    std::vector<std::array<double, 4>> centers(K);
    for (int k = 0; k < K; ++k) {
      centers[static_cast<size_t>(k)] = samples[static_cast<size_t>(rng.below(static_cast<uint32_t>(samples.size())))].x;
    }
    auto nearest = [&](const std::array<double, 4>& x) {
      int best = 0;
      double bd = 1e18;
      for (int k = 0; k < K; ++k) {
        double d = 0;
        for (int j = 0; j < 4; ++j) {
          double diff = x[static_cast<size_t>(j)] - centers[static_cast<size_t>(k)][static_cast<size_t>(j)];
          d += diff * diff;
        }
        if (d < bd) {
          bd = d;
          best = k;
        }
      }
      return best;
    };
    for (int epoch = 0; epoch < 40; ++epoch) {
      std::vector<std::array<double, 4>> sum(K, {0, 0, 0, 0});
      std::vector<int> cnt(K, 0);
      for (const auto& s : samples) {
        int k = nearest(s.x);
        for (int j = 0; j < 4; ++j) sum[static_cast<size_t>(k)][static_cast<size_t>(j)] += s.x[static_cast<size_t>(j)];
        cnt[static_cast<size_t>(k)]++;
      }
      for (int k = 0; k < K; ++k) {
        if (!cnt[static_cast<size_t>(k)]) continue;
        for (int j = 0; j < 4; ++j) {
          centers[static_cast<size_t>(k)][static_cast<size_t>(j)] =
              sum[static_cast<size_t>(k)][static_cast<size_t>(j)] / cnt[static_cast<size_t>(k)];
        }
      }
    }

    // purity + recognition
    std::vector<std::map<int, int>> votes(K);
    std::vector<int> cluster_size(K, 0);
    for (const auto& s : samples) {
      int k = nearest(s.x);
      votes[static_cast<size_t>(k)][s.phone]++;
      cluster_size[static_cast<size_t>(k)]++;
    }
    double purity = 0;
    int total = 0;
    for (int k = 0; k < K; ++k) {
      int bc = 0;
      for (const auto& [p, c] : votes[static_cast<size_t>(k)]) {
        if (c > bc) bc = c;
      }
      purity += bc;
      total += cluster_size[static_cast<size_t>(k)];
    }
    printf("auditory,purity=%.3f\n", total ? purity / total : 0.0);

    int ok = 0, n = 0;
    for (size_t p = 0; p < phones.size(); ++p) {
      Sample s;
      s.phone = static_cast<int>(p);
      for (int d = 0; d < 4; ++d) s.x[static_cast<size_t>(d)] = targets[p][static_cast<size_t>(d)];
      int k = nearest(s.x);
      int best = 0, bc = 0;
      for (const auto& [pp, c] : votes[static_cast<size_t>(k)]) {
        if (c > bc) {
          bc = c;
          best = pp;
        }
      }
      ++n;
      if (best == static_cast<int>(p)) ++ok;
    }
    printf("auditory,recognition_top1=%d/%d\n", ok, n);
  }

  return 0;
}