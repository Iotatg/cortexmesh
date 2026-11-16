// Ordered speech with nested theta-gamma phases, co-articulation, and a
// slow-phase generalisation test. No tokens: phone codes + bit rotation.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/encoder.hpp"
#include "bm/speech.hpp"
#include "bm/sdr.hpp"
#include "bm/words.hpp"

using namespace bm;

namespace {

SDR Rotate(const SDR& s, size_t r) {
  size_t n = s.size();
  if (n == 0) return s;
  r %= n;
  SDR o(n, 0);
  for (uint32_t i : s.indices()) o.set(static_cast<uint32_t>((i + r) % n));
  return o;
}

SDR FirstK(const SDR& s, size_t k) {
  SDR o(s.size(), 0);
  auto idx = s.indices();
  for (size_t i = 0; i < k && i < idx.size(); ++i) o.set(idx[i]);
  return o;
}

// phase offset: nested = slow(syllable) x fast(phone)
size_t Phase(int mode, size_t phone_idx, size_t syll_idx, size_t n) {
  if (mode == 0) return 0;                                  // bag
  if (mode == 1) return (phone_idx % 4) * (n / 8);          // flat
  return (syll_idx % 2) * (n / 2) + (phone_idx % 4) * (n / 16);  // nested
}

bool IsVowel(const std::string& p) {
  static const char* v[] = {"aa", "a", "ii", "u", "ee", "o", "ai", "au", "oo", "uu", "e"};
  for (const char* x : v) {
    if (p == x) return true;
  }
  return false;
}

struct Audio {
  SDR code;
  std::vector<size_t> syll_of;  // syllable index per phone
};

Audio WordAudio(const SpeechModel& sm, const std::vector<std::string>& phones, int mode,
                bool coart, size_t n, size_t k) {
  Audio a;
  a.code = SDR(n, 0);
  size_t syll = 0;
  for (size_t i = 0; i < phones.size(); ++i) {
    if (i > 0 && IsVowel(phones[i - 1]) && !IsVowel(phones[i])) ++syll;
    a.syll_of.push_back(syll);
    SDR ph = coart && i > 0 ? sm.acoustic(phones[i], phones[i - 1]) : sm.acoustic(phones[i]);
    a.code |= Rotate(ph, Phase(mode, i, syll, n));
  }
  a.code = FirstK(a.code, k);
  return a;
}

int RankOf(const AssociativeMemory& m, const SDR& in, const SDR& target) {
  auto ranked = m.scores(in);
  std::vector<char> t(target.size(), 0);
  for (uint32_t i : target.indices()) t[i] = 1;
  for (size_t r = 0; r < ranked.size(); ++r) {
    if (t[ranked[r].first]) return static_cast<int>(r);
  }
  return 1 << 20;
}

}  // namespace

int main() {
  const size_t N = 1024, K = 24;
  WordEncoder we(512, 24, 11);
  SpeechModel sm(N, K, 31, true);
  sm.train(60);

  std::vector<std::string> words;
  for (const auto& w : we.vocabulary()) {
    if (w == "<s>" || w == "</s>") continue;
    words.push_back(w);
  }
  const std::string held = "pyaas";
  std::vector<std::string> train_words;
  for (const auto& w : words) {
    if (w != held) train_words.push_back(w);
  }

  for (int mode = 0; mode < 3; ++mode) {
    const char* name = mode == 0 ? "bag" : (mode == 1 ? "flat4" : "nested");
    for (int coart = 0; coart < 2; ++coart) {
      std::map<std::string, Audio> audio;
      std::map<std::string, SDR> word_code;
      for (const auto& w : train_words) {
        const WordEntry* e = we.entry(w);
        if (!e) continue;
        audio[w] = WordAudio(sm, e->phones, mode, coart != 0, N, K);
        word_code[w] = we.encode(w);
      }
      AssociativeMemory hear(N, 512, 24, 0.05f);  // audio -> word
      AssociativeMemory say(512, N, 24, 0.05f);   // word -> audio
      for (int ep = 0; ep < 6; ++ep) {
        for (const auto& w : train_words) {
          if (!audio.count(w)) continue;
          hear.learn(audio[w].code, word_code[w]);
          say.learn(word_code[w], audio[w].code);
        }
      }

      // hear
      int h1 = 0, h3 = 0, n = 0;
      for (const auto& w : train_words) {
        if (!audio.count(w)) continue;
        int r = RankOf(hear, audio[w].code, word_code[w]);
        ++n;
        h1 += r == 0;
        h3 += r < 3;
      }
      // say: word -> predicted audio -> hear
      int s1 = 0, s3 = 0, sn = 0;
      double say_overlap = 0;
      for (const auto& w : train_words) {
        if (!audio.count(w)) continue;
        SDR pred = say.recall(word_code[w]);
        say_overlap += float(audio[w].code.overlap(pred)) / float(K);
        int r = RankOf(hear, pred, word_code[w]);
        ++sn;
        s1 += r == 0;
        s3 += r < 3;
      }
      // reversal
      double rev = 0;
      int rn = 0;
      for (const auto& w : train_words) {
        if (!audio.count(w)) continue;
        const WordEntry* e = we.entry(w);
        std::vector<std::string> revp(e->phones.rbegin(), e->phones.rend());
        Audio ra = WordAudio(sm, revp, mode, coart != 0, N, K);
        rev += float(audio[w].code.overlap(ra.code)) / float(K);
        ++rn;
      }
      // co-articulation sensitivity: same word, isolated vs context
      double ctx_overlap = 0;
      int cn = 0;
      for (const auto& w : train_words) {
        if (!audio.count(w)) continue;
        const WordEntry* e = we.entry(w);
        Audio iso = WordAudio(sm, e->phones, mode, false, N, K);
        ctx_overlap += float(audio[w].code.overlap(iso.code)) / float(K);
        ++cn;
      }

      printf("%-7s coart=%d hear=%d/%d(%.3f) say=%d/%d(%.3f) sayov=%.3f rev=%.3f ctx=%.3f\n",
             name, coart, h1, n, n ? double(h3) / n : 0, s1, sn, sn ? double(s3) / sn : 0,
             sn ? say_overlap / sn : 0, rn ? rev / rn : 0, cn ? ctx_overlap / cn : 0);
    }
  }

  // held-out word: rank among all trained words
  {
    const WordEntry* e = we.entry(held);
    std::map<std::string, SDR> word_code;
    AssociativeMemory hear(N, 512, 24, 0.05f);
    for (const auto& w : train_words) {
      const WordEntry* we2 = we.entry(w);
      if (!we2) continue;
      Audio a = WordAudio(sm, we2->phones, 2, false, N, K);
      word_code[w] = we.encode(w);
      for (int ep = 0; ep < 6; ++ep) hear.learn(a.code, word_code[w]);
    }
    Audio h = WordAudio(sm, e->phones, 2, false, N, K);
    auto ranked = hear.scores(h.code);
    int rank = -1;
    for (size_t r = 0; r < ranked.size() && r < 50; ++r) {
      bool hit = false;
      for (uint32_t i : we.encode(held).indices()) {
        if (ranked[r].first == i) hit = true;
      }
      if (hit) {
        rank = static_cast<int>(r);
        break;
      }
    }
    printf("heldout,%s,rank=%d,of=%zu\n", held.c_str(), rank, train_words.size());
  }

  // slow-phase generalisation: same syllable, different position
  {
    // collect slow-phase half of each word's audio, group by syllable code
    std::map<std::string, std::vector<SDR>> by_syll;
    for (const auto& w : train_words) {
      const WordEntry* e = we.entry(w);
      if (!e) continue;
      Audio a = WordAudio(sm, e->phones, 2, false, N, K);
      for (size_t sy = 0; sy < 2; ++sy) {
        SDR half(N, 0);
        for (uint32_t i : a.code.indices()) {
          if (i < N / 2) half.set(i);
        }
        // label: the syllable that occupies slow slot sy
        std::string label = "?";
        for (size_t i = 0; i < e->phones.size(); ++i) {
          if (a.syll_of[i] == sy) {
            label = e->phones[i];
            break;
          }
        }
        if (label != "?") by_syll[label].push_back(half);
      }
    }
    double same = 0, diff = 0;
    int ns = 0, nd = 0;
    for (const auto& [lab, codes] : by_syll) {
      for (size_t i = 0; i < codes.size(); ++i) {
        for (size_t j = i + 1; j < codes.size(); ++j) {
          same += float(codes[i].overlap(codes[j])) / float(K);
          ++ns;
        }
      }
      (void)lab;
    }
    std::vector<std::string> labs;
    for (const auto& [lab, codes] : by_syll) labs.push_back(lab);
    for (size_t i = 0; i < labs.size(); ++i) {
      for (size_t j = i + 1; j < labs.size(); ++j) {
        for (const auto& a : by_syll[labs[i]]) {
          for (const auto& b : by_syll[labs[j]]) {
            diff += float(a.overlap(b)) / float(K);
            ++nd;
          }
        }
      }
    }
    printf("slowphase,same=%.3f,diff=%.3f,n_same=%d,n_diff=%d\n", ns ? same / ns : 0,
           nd ? diff / nd : 0, ns, nd);
  }

  return 0;
}