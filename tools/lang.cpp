#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/interoception.hpp"
#include "bm/sdr.hpp"
#include "bm/words.hpp"

using namespace bm;

namespace {

SDR Shift(const SDR& s, size_t off) {
  SDR o(s.size(), 0);
  for (uint32_t i : s.indices()) o.set(static_cast<uint32_t>((i + off) % s.size()));
  return o;
}

SDR Ctx(const SDR& prev2, const SDR& prev1) {
  return Shift(prev2, 0) | Shift(prev1, prev1.size() / 2);
}

SDR Noise(const SDR& s, int flips, Rng& rng) {
  SDR o = s;
  for (int i = 0; i < flips; ++i) {
    auto idx = s.indices();
    if (idx.empty()) break;
    o.reset(idx[rng.below(static_cast<uint32_t>(idx.size()))]);
  }
  for (int i = 0; i < flips; ++i) {
    for (int guard = 0; guard < 64; ++guard) {
      uint32_t bit = rng.below(static_cast<uint32_t>(s.size()));
      if (!o.test(bit)) {
        o.set(bit);
        break;
      }
    }
  }
  return o;
}

std::vector<std::pair<std::string, float>> Rank(const SDR& r, const WordEncoder& we) {
  std::vector<std::pair<std::string, float>> v;
  for (const auto& w : we.vocabulary()) {
    v.emplace_back(w, static_cast<float>(we.encode(w).overlap(r)));
  }
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  return v;
}

struct Acc {
  int top1 = 0, top3 = 0, n = 0;
  double rr = 0.0;
};



class WordReadout {
 public:
  WordReadout(size_t n, const std::vector<std::string>& vocab, const WordEncoder& we)
      : n_(n), vocab_(vocab), we_(we) {
    for (size_t j = 0; j < vocab_.size(); ++j) w_.emplace_back(n_, 0.0f);
  }

  std::vector<std::pair<std::string, float>> scores(const SDR& ctx) const {
    std::vector<std::pair<std::string, float>> v;
    v.reserve(vocab_.size());
    for (size_t j = 0; j < vocab_.size(); ++j) {
      float s = 0.0f;
      for (uint32_t i : ctx.indices()) s += w_[j][i];
      v.emplace_back(vocab_[j], s);
    }
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return v;
  }

  void learn(const SDR& ctx, const std::string& correct, float lr) {
    float scale = ctx.count() ? 1.0f / static_cast<float>(ctx.count()) : 0.0f;
    for (size_t j = 0; j < vocab_.size(); ++j) {
      float s = 0.0f;
      for (uint32_t i : ctx.indices()) s += w_[j][i];
      float target = (vocab_[j] == correct) ? 1.0f : 0.0f;
      float err = target - s;
      if (std::fabs(err) < 1e-4f) continue;
      float step = lr * err * scale;
      for (uint32_t i : ctx.indices()) w_[j][i] += step;
    }
  }

 private:
  size_t n_;
  std::vector<std::string> vocab_;
  const WordEncoder& we_;
  std::vector<std::vector<float>> w_;
};

Acc EvalReadout(const WordReadout& ro, const WordEncoder& we,
                const std::vector<std::vector<std::string>>& sentences, bool train, float lr) {
  Acc a;
  WordReadout* mut = nullptr;
  (void)mut;
  for (const auto& s : sentences) {
    SDR p2 = we.encode("<s>");
    SDR p1 = we.encode("<s>");
    for (const auto& w : s) {
      SDR ctx = Ctx(p2, p1);
      auto ranked = ro.scores(ctx);
      ++a.n;
      int rank = -1;
      for (size_t i = 0; i < ranked.size(); ++i) {
        if (ranked[i].first == w) {
          rank = static_cast<int>(i) + 1;
          break;
        }
      }
      if (rank == 1) ++a.top1;
      if (rank >= 1 && rank <= 3) ++a.top3;
      if (rank >= 1) a.rr += 1.0 / rank;
      p2 = p1;
      p1 = we.encode(w);
    }
    if (train) {
      // second pass to update
      p2 = we.encode("<s>");
      p1 = we.encode("<s>");
      for (const auto& w : s) {
        const_cast<WordReadout&>(ro).learn(Ctx(p2, p1), w, lr);
        p2 = p1;
        p1 = we.encode(w);
      }
      const_cast<WordReadout&>(ro).learn(Ctx(p2, p1), "</s>", lr);
    }
  }
  return a;
}

}  // namespace

int main() {
  WordEncoder we(512, 24, 11);

  std::vector<std::vector<std::string>> sentences = {
      {"main", "bhookha", "hoon"},
      {"main", "thak", "gaya", "hoon"},
      {"main", "khush", "hoon"},
      {"main", "theek", "hoon"},
      {"mujhe", "paani", "chahiye"},
      {"mujhe", "dard", "ho", "raha", "hai"},
      {"mujhe", "neend", "aa", "rahi", "hai"},
      {"mujhe", "bhookh", "lagi", "hai"},
      {"tum", "kaise", "ho"},
      {"shukriya"},
      {"mujhe", "madad", "chahiye"},
  };
  const size_t n_train = sentences.size() - 2;
  std::vector<std::vector<std::string>> train(sentences.begin(), sentences.begin() + n_train);
  std::vector<std::vector<std::string>> test(sentences.begin() + n_train, sentences.end());

  // 1. word codes are feature bundles: similar words overlap more
  {
    double same = 0, diff = 0;
    int ns = 0, nd = 0;
    const auto vocab = we.vocabulary();
    for (size_t i = 0; i < vocab.size(); ++i) {
      for (size_t j = i + 1; j < vocab.size(); ++j) {
        const WordEntry* a = we.entry(vocab[i]);
        const WordEntry* b = we.entry(vocab[j]);
        if (!a || !b) continue;
        size_t shared = 0;
        for (const auto& p : a->phones) {
          for (const auto& q : b->phones) {
            if (p == q) ++shared;
          }
        }
        double ov = static_cast<double>(we.encode(vocab[i]).overlap(we.encode(vocab[j])));
        if (shared >= 2) {
          same += ov;
          ++ns;
        } else if (shared == 0) {
          diff += ov;
          ++nd;
        }
      }
    }
    printf("similarity,same_phone=%.2f,unrelated=%.2f,ratio=%.2f\n", ns ? same / ns : 0.0,
           nd ? diff / nd : 0.0, (nd && diff > 0) ? (same / ns) / (diff / nd) : 0.0);
  }

  // context ambiguity: how many distinct continuations share a context
  {
    std::unordered_map<std::string, std::vector<std::string>> nxt;
    for (const auto& s : sentences) {
      std::vector<std::string> full;
      full.push_back("<s>");
      full.push_back("<s>");
      for (const auto& w : s) full.push_back(w);
      full.push_back("</s>");
      for (size_t i = 2; i < full.size(); ++i) {
        std::string key = full[i - 2] + "\x01" + full[i - 1];
        auto& v = nxt[key];
        if (std::find(v.begin(), v.end(), full[i]) == v.end()) v.push_back(full[i]);
      }
    }
    size_t amb = 0, multi = 0;
    for (const auto& [k, v] : nxt) {
      ++amb;
      if (v.size() > 1) ++multi;
    }
    printf("context,total=%zu,ambiguous=%zu,ambiguity_rate=%.3f\n", amb, multi,
           amb ? static_cast<double>(multi) / amb : 0.0);
  }

  // 2. sequence learning: context is a bundle of previous word codes (no token ids)
  WordReadout ro(512, we.vocabulary(), we);
  for (int epoch = 0; epoch < 60; ++epoch) {
    EvalReadout(ro, we, train, true, 0.05f);
  }
  Acc tr = EvalReadout(ro, we, train, false, 0.0f);
  Acc te = EvalReadout(ro, we, test, false, 0.0f);
  printf("sequence,train_top1=%.3f,train_top3=%.3f,train_mrr=%.3f,test_top1=%.3f,test_top3=%.3f,test_mrr=%.3f\n",
         tr.n ? static_cast<double>(tr.top1) / tr.n : 0.0,
         tr.n ? static_cast<double>(tr.top3) / tr.n : 0.0,
         tr.n ? tr.rr / tr.n : 0.0, te.n ? static_cast<double>(te.top1) / te.n : 0.0,
         te.n ? static_cast<double>(te.top3) / te.n : 0.0, te.n ? te.rr / te.n : 0.0);

  // novel word: same phones as "paani" but unseen in training
  {
    auto ranked = ro.scores(Ctx(we.encode("<s>"), we.encode("mujhe")));
    int rank_pyaas = -1;
    for (size_t i = 0; i < ranked.size(); ++i) {
      if (ranked[i].first == "pyaas") rank_pyaas = static_cast<int>(i) + 1;
    }
    printf("novelty,after_mujhe_top=%s,rank_pyaas=%d\n", ranked[0].first.c_str(), rank_pyaas);
  }

  // 3. grounding: word <-> grounded features
  {
    AssociativeMemory w2c(512, 512, 24, 0.08f);
    AssociativeMemory c2w(512, 512, 24, 0.08f);
    for (int epoch = 0; epoch < 6; ++epoch) {
      for (const auto& w : we.vocabulary()) {
        const WordEntry* e = we.entry(w);
        if (!e || e->concepts.empty()) continue;
        SDR cc(512, 0);
        for (const auto& c : e->concepts) cc |= we.concept_code(c);
        w2c.learn(we.encode(w), cc);
        c2w.learn(cc, we.encode(w));
      }
    }
    int ok_w2c = 0, n_w2c = 0, ok_c2w = 0, n_c2w = 0;
    for (const auto& w : we.vocabulary()) {
      const WordEntry* e = we.entry(w);
      if (!e || e->concepts.empty()) continue;
      SDR cc(512, 0);
      for (const auto& c : e->concepts) cc |= we.concept_code(c);
      SDR r = w2c.recall(we.encode(w));
      bool hit = false;
      for (const auto& c : e->concepts) {
        if (we.concept_code(c).overlap(r) > 0) hit = true;
      }
      ++n_w2c;
      if (hit) ++ok_w2c;
      SDR r2 = c2w.recall(cc);
      auto ranked = Rank(r2, we);
      ++n_c2w;
      if (!ranked.empty() && ranked[0].first == w) ++ok_c2w;
    }
    printf("grounding,word_to_concept=%.3f,concept_to_word=%.3f\n",
           n_w2c ? static_cast<double>(ok_w2c) / n_w2c : 0.0,
           n_c2w ? static_cast<double>(ok_c2w) / n_c2w : 0.0);
  }

  // 4. feeling: body state -> word (interoception)
  {
    Interoception io(512);
    std::vector<std::pair<std::string, BodyState>> feelings = {
        {"bhookh", {92, 34, 15, 0.25f}},
        {"pyaas", {100, 30, 17, 0.35f}},
        {"thak", {68, 58, 12, 0.75f}},
        {"dard", {112, 26, 21, 0.85f}},
        {"khush", {76, 72, 14, 0.20f}},
    };
    AssociativeMemory feel(512, 512, 24, 0.10f);
    for (int epoch = 0; epoch < 40; ++epoch) {
      for (const auto& f : feelings) feel.learn(io.encode(f.second), we.encode(f.first));
    }
    Rng rng(21);
    for (int flips : {0, 2, 4, 8}) {
      int ok = 0;
      for (const auto& f : feelings) {
        SDR body = io.encode(f.second);
        if (flips) body = Noise(body, flips, rng);
        auto ranked = Rank(feel.recall(body), we);
        if (!ranked.empty() && ranked[0].first == f.first) ++ok;
      }
      printf("feeling,flips=%d,top1=%.3f\n", flips, static_cast<double>(ok) / feelings.size());
    }
  }

  // 5. generation from prediction
  {
    SDR p2 = we.encode("<s>");
    SDR p1 = we.encode("<s>");
    std::string line = "dialogue,";
    std::string prev_word;
    for (int step = 0; step < 8; ++step) {
      auto ranked = ro.scores(Ctx(p2, p1));
      if (ranked.empty()) break;
      const std::string& w = ranked[0].first;
      if (w == "</s>" || w == "<s>") break;
      if (step >= 2 && w == prev_word) break;
      prev_word = w;
      if (step) line += " ";
      line += w;
      p2 = p1;
      p1 = we.encode(w);
    }
    printf("%s\n", line.c_str());
  }

  return 0;
}
