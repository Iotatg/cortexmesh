#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/sdr.hpp"
#include "bm/speech.hpp"
#include "bm/words.hpp"

using namespace bm;

namespace {

SDR WordAudio(const SpeechModel& sm, const std::vector<std::string>& phones) {
  SDR a(sm.size(), 0);
  for (const auto& p : phones) a |= sm.acoustic(p);
  return a;
}

bool Covered(const SpeechModel& sm, const WordEntry& e) {
  for (const auto& p : e.phones) {
    bool ok = false;
    for (const auto& q : sm.phones()) {
      if (q == p) ok = true;
    }
    if (!ok) return false;
  }
  return true;
}

int RankOf(const SDR& r, const std::vector<SDR>& items, size_t target) {
  std::vector<float> ovs;
  ovs.reserve(items.size());
  for (const auto& it : items) ovs.push_back(static_cast<float>(it.overlap(r)));
  float target_ov = ovs[target];
  int rank = 1;
  for (size_t i = 0; i < ovs.size(); ++i) {
    if (i != target && ovs[i] > target_ov) ++rank;
  }
  return rank;
}

}  // namespace

int main() {
  WordEncoder we(1024, 24, 11);
  SpeechModel sm(1024, 24, 31, true);
  sm.train(200);

  std::vector<std::string> words;
  std::vector<SDR> word_code;
  std::vector<SDR> word_audio;
  std::vector<std::vector<std::string>> phones;
  for (const auto& w : we.vocabulary()) {
    const WordEntry* e = we.entry(w);
    if (!e || !Covered(sm, *e)) continue;
    words.push_back(w);
    word_code.push_back(we.encode(w));
    phones.push_back(e->phones);
    word_audio.push_back(WordAudio(sm, e->phones));
  }
  printf("words,covered=%zu\n", words.size());

  // word-level loop: hear word -> word, say word -> hear word
  AssociativeMemory a2w(1024, 1024, 24, 0.06f);
  AssociativeMemory w2a(1024, 1024, 96, 0.06f);
  for (int e = 0; e < 200; ++e) {
    for (size_t i = 0; i < words.size(); ++i) {
      a2w.learn(word_audio[i], word_code[i]);
      w2a.learn(word_code[i], word_audio[i]);
    }
  }

  int h1 = 0, h3 = 0, s1 = 0, s3 = 0;
  for (size_t i = 0; i < words.size(); ++i) {
    int hr = RankOf(a2w.recall(word_audio[i]), word_code, i);
    int sr = RankOf(w2a.recall(word_code[i]), word_audio, i);
    if (hr == 1) ++h1;
    if (hr <= 3) ++h3;
    if (sr == 1) ++s1;
    if (sr <= 3) ++s3;
  }
  printf("word_loop,hear_top1=%d/%zu,hear_top3=%d,say_top1=%d/%zu,say_top3=%d\n", h1,
         words.size(), h3, s1, words.size(), s3);

  // held-out word: train without "pyaas", then hear it
  {
    AssociativeMemory ho(1024, 1024, 24, 0.06f);
    int held = -1;
    for (size_t i = 0; i < words.size(); ++i) {
      if (words[i] == "pyaas") held = static_cast<int>(i);
    }
    for (int e = 0; e < 200; ++e) {
      for (size_t i = 0; i < words.size(); ++i) {
        if (static_cast<int>(i) == held) continue;
        ho.learn(word_audio[i], word_code[i]);
      }
    }
    if (held >= 0) {
      int hr = RankOf(ho.recall(word_audio[static_cast<size_t>(held)]), word_code,
                      static_cast<size_t>(held));
      printf("word_heldout,pyaas_rank=%d/%zu\n", hr, words.size());
    }
  }


  // ordered audio: phase-bind phone positions (theta-gamma)
  {
    auto seq = [&](const std::vector<std::string>& ph) {
      SDR a(sm.size(), 0);
      for (size_t i = 0; i < ph.size(); ++i) {
        size_t sh = (i % 4) * (sm.size() / 4);
        SDR c = sm.acoustic(ph[i]);
        SDR t(sm.size(), 0);
        for (uint32_t b : c.indices()) t.set(static_cast<uint32_t>((b + sh) % sm.size()));
        a |= t;
      }
      return a;
    };
    std::vector<SDR> seq_audio;
    double rev_ov = 0;
    for (size_t i = 0; i < phones.size(); ++i) {
      seq_audio.push_back(seq(phones[i]));
      std::vector<std::string> rev(phones[i].rbegin(), phones[i].rend());
      rev_ov += static_cast<double>(seq_audio.back().overlap(seq(rev)));
    }

    AssociativeMemory sa(1024, 1024, 24, 0.06f);
    AssociativeMemory sw(1024, 1024, 96, 0.06f);
    for (int e = 0; e < 200; ++e) {
      for (size_t i = 0; i < words.size(); ++i) {
        sa.learn(seq_audio[i], word_code[i]);
        sw.learn(word_code[i], seq_audio[i]);
      }
    }
    int h1 = 0, h3 = 0, s1 = 0, s3 = 0;
    for (size_t i = 0; i < words.size(); ++i) {
      int hr = RankOf(sa.recall(seq_audio[i]), word_code, i);
      int sr = RankOf(sw.recall(word_code[i]), seq_audio, i);
      if (hr == 1) ++h1;
      if (hr <= 3) ++h3;
      if (sr == 1) ++s1;
      if (sr <= 3) ++s3;
    }
    printf("sequence,hear_top1=%d/%zu,hear_top3=%d,say_top1=%d,say_top3=%d,reversal_overlap=%.1f\n",
           h1, words.size(), h3, s1, s3, words.size() ? rev_ov / words.size() : 0.0);
  }


  // syllable layer: word -> syllables -> phones, then phase-bind both levels
  {
    auto is_vowel = [](const std::string& p) {
      static const char* v[] = {"aa", "a", "ii", "u", "ee", "o", "ai", "au", "oo", "uu", "e"};
      for (const char* x : v) {
        if (p == x) return true;
      }
      return false;
    };
    auto syllabify = [&](const std::vector<std::string>& ph) {
      std::vector<std::vector<std::string>> out;
      std::vector<std::string> cur;
      for (const auto& p : ph) {
        cur.push_back(p);
        if (is_vowel(p)) {
          out.push_back(cur);
          cur.clear();
        }
      }
      if (!cur.empty()) {
        if (!out.empty()) {
          out.back().insert(out.back().end(), cur.begin(), cur.end());
        } else {
          out.push_back(cur);
        }
      }
      return out;
    };
    auto bind = [&](const std::vector<std::string>& ph, size_t level) {
      SDR a(sm.size(), 0);
      for (size_t i = 0; i < ph.size(); ++i) {
        size_t sh = ((i + level) % 4) * (sm.size() / 4);
        SDR c = sm.acoustic(ph[i]);
        SDR t(sm.size(), 0);
        for (uint32_t b : c.indices()) t.set(static_cast<uint32_t>((b + sh) % sm.size()));
        a |= t;
      }
      return a;
    };
    std::vector<SDR> syl_audio;
    size_t syl_total = 0;
    for (size_t i = 0; i < phones.size(); ++i) {
      auto sylls = syllabify(phones[i]);
      syl_total += sylls.size();
      SDR a(sm.size(), 0);
      for (size_t j = 0; j < sylls.size(); ++j) {
        size_t sh = (j % 4) * (sm.size() / 4);
        SDR sc = bind(sylls[j], 0);
        SDR t(sm.size(), 0);
        for (uint32_t b : sc.indices()) t.set(static_cast<uint32_t>((b + sh) % sm.size()));
        a |= t;
      }
      syl_audio.push_back(a);
    }
    AssociativeMemory sa(1024, 1024, 24, 0.06f);
    AssociativeMemory sw(1024, 1024, 96, 0.06f);
    for (int e = 0; e < 200; ++e) {
      for (size_t i = 0; i < words.size(); ++i) {
        sa.learn(syl_audio[i], word_code[i]);
        sw.learn(word_code[i], syl_audio[i]);
      }
    }
    int h1 = 0, s1 = 0;
    for (size_t i = 0; i < words.size(); ++i) {
      if (RankOf(sa.recall(syl_audio[i]), word_code, i) == 1) ++h1;
      if (RankOf(sw.recall(word_code[i]), syl_audio, i) == 1) ++s1;
    }
    printf("syllable,words=%zu,syllables=%zu,hear_top1=%d,say_top1=%d\n", words.size(),
           syl_total, h1, s1);
  }

  // co-articulation: the same phone sounds different after different neighbours
  {
    const char* pairs[][2] = {{"k", "aa"}, {"t", "aa"}, {"p", "aa"}};
    for (const auto& pr : pairs) {
      SDR a = sm.acoustic(pr[0]);
      SDR b = sm.acoustic(pr[0], pr[1]);
      printf("coarticulation,phone=%s,self_overlap=%.1f,context_overlap=%.1f\n", pr[0],
             static_cast<double>(a.overlap(b)),
             static_cast<double>(sm.acoustic("aa").overlap(b)));
    }
  }

  // context-aware inverse: input = prev + current articulation -> current sound
  {
    AssociativeMemory ctx(1024, 1024, 96, 0.06f);
    std::vector<std::string> prevs = {"p", "t", "k", "aa", "ii"};
    for (int e = 0; e < 200; ++e) {
      for (const auto& pv : prevs) {
        for (const auto& p : sm.phones()) {
          SDR in = sm.articulatory(pv) | sm.articulatory(p);
          ctx.learn(in, sm.acoustic(p, pv));
        }
      }
    }
    int ok = 0, n = 0;
    for (const auto& pv : prevs) {
      for (const auto& p : sm.phones()) {
        SDR in = sm.articulatory(pv) | sm.articulatory(p);
        SDR r = ctx.recall(in);
        int best = -1;
        float best_ov = -1.0f;
        for (const auto& q : sm.phones()) {
          float ov = static_cast<float>(sm.acoustic(q, pv).overlap(r));
          if (ov > best_ov) {
            best_ov = ov;
            best = static_cast<int>(&q - &sm.phones()[0]);
          }
        }
        ++n;
        if (best >= 0 && sm.phones()[static_cast<size_t>(best)] == p) ++ok;
      }
    }
    printf("context_inverse,top1=%d/%d\n", ok, n);
  }

  return 0;
}