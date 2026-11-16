// Train the SDR language layer on a real corpus fetched from Hugging Face.
// Word = cached sparse code (no integer ids). Context = phase-shifted bundle of
// the previous two words. Two readouts are compared:
//   (a) one Hebbian matrix, decoded by normalised bit scores
//   (b) a sparse delta-rule readout (one weight vector per word, local updates)
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/encoder.hpp"
#include "bm/sdr.hpp"

using namespace bm;

namespace {

std::string Norm(const std::string& w) {
  std::string o;
  for (char c : w) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'') {
      o.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return o;
}

std::vector<std::string> Tokenize(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'') {
      cur.push_back(c);
    } else if (!cur.empty()) {
      out.push_back(Norm(cur));
      cur.clear();
    }
  }
  if (!cur.empty()) out.push_back(Norm(cur));
  return out;
}

SDR Rotate(const SDR& s, size_t r) {
  size_t n = s.size();
  if (!n) return s;
  r %= n;
  SDR o(n, 0);
  for (uint32_t i : s.indices()) o.set(static_cast<uint32_t>((i + r) % n));
  return o;
}

struct Eval {
  int n = 0, h1 = 0, h3 = 0, h10 = 0, inv_n = 0, inv_h1 = 0, inv_h10 = 0;
  double mrr = 0, oov = 0;
  void add(int rank, bool out_of_vocab) {
    ++n;
    h1 += rank == 0;
    h3 += rank < 3;
    h10 += rank < 10;
    if (!out_of_vocab) {
      ++inv_n;
      inv_h1 += rank == 0;
      inv_h10 += rank < 10;
    }
    mrr += 1.0 / (1.0 + std::max(0, rank));
    oov += out_of_vocab ? 1 : 0;
  }
  void report(const char* tag) const {
    printf("%-18s n=%6d top1=%.4f top3=%.4f top10=%.4f mrr=%.4f | inv top1=%.4f top10=%.4f | "
           "oov=%.4f\n",
           tag, n, n ? double(h1) / n : 0, n ? double(h3) / n : 0, n ? double(h10) / n : 0,
           n ? mrr / n : 0, inv_n ? double(inv_h1) / inv_n : 0,
           inv_n ? double(inv_h10) / inv_n : 0, n ? oov / n : 0);
  }
};

// sparse delta-rule readout: one weight vector per word, updates touch only the
// correct word and the current top competitors
struct Readout {
  size_t N = 0, V = 0;
  float lr = 0.05f;
  int top = 8;
  std::vector<float> w;
  std::vector<std::vector<int>> bit_to_words;
  std::vector<float> scr;
  std::vector<char> seen;
  std::vector<int> touched;

  void build(size_t n, const std::vector<SDR>& codes) {
    N = n;
    V = codes.size();
    w.assign(V * N, 0.0f);
    bit_to_words.assign(N, {});
    for (size_t v = 0; v < V; ++v) {
      for (uint32_t i : codes[v].indices()) bit_to_words[i].push_back(static_cast<int>(v));
    }
    scr.assign(V, 0.0f);
    seen.assign(V, 0);
    touched.reserve(4096);
  }

  void clear() {
    for (int v : touched) {
      scr[static_cast<size_t>(v)] = 0.0f;
      seen[static_cast<size_t>(v)] = 0;
    }
    touched.clear();
  }

  void accumulate(const SDR& ctx) {
    clear();
    for (uint32_t i : ctx.indices()) {
      for (int v : bit_to_words[i]) {
        size_t vv = static_cast<size_t>(v);
        if (!seen[vv]) {
          seen[vv] = 1;
          touched.push_back(v);
        }
        scr[vv] += w[vv * N + i];
      }
    }
  }

  void train(const std::vector<std::pair<SDR, int>>& data, int epochs) {
    std::vector<std::pair<float, int>> cand;
    for (int e = 0; e < epochs; ++e) {
      for (const auto& [ctx, target] : data) {
        accumulate(ctx);
        float st = scr[static_cast<size_t>(target)];
        cand.clear();
        for (int v : touched) {
          cand.emplace_back(scr[static_cast<size_t>(v)], v);
        }
        int k = std::min<int>(top, static_cast<int>(cand.size()));
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        float* wt = w.data() + static_cast<size_t>(target) * N;
        for (uint32_t i : ctx.indices()) wt[i] += lr * (1.0f - st);
        for (int j = 0; j < k; ++j) {
          int v = cand[static_cast<size_t>(j)].second;
          if (v == target) continue;
          float sv = cand[static_cast<size_t>(j)].first;
          if (sv <= 0.0f) continue;
          float* wv = w.data() + static_cast<size_t>(v) * N;
          for (uint32_t i : ctx.indices()) wv[i] -= lr * sv;
        }
      }
    }
  }

  int rank(const SDR& ctx, int target) {
    accumulate(ctx);
    float tv = scr[static_cast<size_t>(target)];
    int above = 0;
    for (int v : touched) {
      if (v != target && scr[static_cast<size_t>(v)] > tv) ++above;
    }
    return above;
  }
};

}  // namespace

size_t PeakRssMb() {
  std::ifstream st("/proc/self/status");
  std::string line;
  while (std::getline(st, line)) {
    if (line.rfind("VmHWM:", 0) == 0) return static_cast<size_t>(std::stol(line.substr(6)) / 1024);
  }
  printf("peak_rss=%zuMB\n", PeakRssMb());
  return 0;
}

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "data/wikitext2.txt";
  const size_t V = argc > 2 ? std::stoul(argv[2]) : 2000;
  const int epochs = argc > 3 ? std::stoi(argv[3]) : 2;
  const size_t N = 1024, K = 32;

  std::ifstream in(path);
  if (!in) {
    printf("cannot open %s\n", path.c_str());
    return 1;
  }
  std::vector<std::vector<std::string>> sents;
  std::string line;
  std::map<std::string, int> freq;
  size_t n_tokens = 0;
  while (std::getline(in, line)) {
    auto toks = Tokenize(line);
    if (toks.size() < 3) continue;
    n_tokens += toks.size();
    sents.push_back(std::move(toks));
    for (const auto& t : sents.back()) freq[t]++;
  }
  printf("corpus=%s sentences=%zu tokens=%zu types=%zu vocab=%zu\n", path.c_str(), sents.size(),
         n_tokens, freq.size(), V);

  std::vector<std::pair<std::string, int>> by_freq(freq.begin(), freq.end());
  std::sort(by_freq.begin(), by_freq.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  std::vector<std::string> vocab_list;
  std::map<std::string, int> vocab_index;
  for (size_t i = 0; i < std::min(V, by_freq.size()); ++i) {
    vocab_index[by_freq[i].first] = static_cast<int>(vocab_list.size());
    vocab_list.push_back(by_freq[i].first);
  }
  const int UNK = static_cast<int>(vocab_list.size());
  vocab_list.push_back("<unk>");

  Encoder enc(N, K, 5);
  std::vector<SDR> word_codes;
  word_codes.reserve(vocab_list.size());
  for (const auto& w : vocab_list) word_codes.push_back(enc.symbol(w));
  auto code_of = [&](const std::string& w) {
    auto it = vocab_index.find(w);
    return it == vocab_index.end() ? UNK : it->second;
  };

  size_t split = sents.size() - std::max<size_t>(1, sents.size() * 15 / 100);
  std::vector<std::vector<std::string>> train(sents.begin(), sents.begin() + split);
  std::vector<std::vector<std::string>> test(sents.begin() + split, sents.end());

  auto ctx_of = [&](int p2, int p1) {
    SDR c = Rotate(word_codes[static_cast<size_t>(p2)], 0);
    c |= Rotate(word_codes[static_cast<size_t>(p1)], N / 2);
    return c;
  };

  std::vector<std::pair<SDR, int>> data;
  for (const auto& s : train) {
    int p2 = 0, p1 = 0;
    for (const auto& w : s) {
      int cur = code_of(w);
      data.emplace_back(ctx_of(p2, p1), cur);
      p2 = p1;
      p1 = cur;
    }
  }
  printf("train_pairs=%zu\n", data.size());

  Readout ro;
  ro.build(N, word_codes);
  auto t0 = std::chrono::steady_clock::now();
  ro.train(data, epochs);
  auto t1 = std::chrono::steady_clock::now();
  printf("readout_seconds=%.2f\n", std::chrono::duration<double>(t1 - t0).count());
  data.clear();
  data.shrink_to_fit();

  AssociativeMemory mat(N, N, K, 0.02f);
  std::vector<std::pair<SDR, SDR>> matrix_pairs;
  for (const auto& s : train) {
    int p2 = 0, p1 = 0;
    for (const auto& w : s) {
      int cur = code_of(w);
      matrix_pairs.emplace_back(ctx_of(p2, p1), word_codes[static_cast<size_t>(cur)]);
      p2 = p1;
      p1 = cur;
    }
  }
  for (int e = 0; e < epochs; ++e) mat.learn_batch(matrix_pairs);
  matrix_pairs.clear();
  matrix_pairs.shrink_to_fit();
  std::vector<double> bit_freq(N, 0.0);
  for (const auto& [i_, o_] : matrix_pairs) {
    (void)i_;
    for (uint32_t j : o_.indices()) bit_freq[j] += 1.0;
  }
  std::vector<std::vector<int>> bit_to_words(N);
  for (size_t v = 0; v < word_codes.size(); ++v) {
    for (uint32_t i : word_codes[v].indices()) bit_to_words[i].push_back(static_cast<int>(v));
  }

  auto evaluate = [&](bool use_readout) {
    Eval ev;
    for (const auto& s : test) {
      int p2 = 0, p1 = 0;
      for (const auto& w : s) {
        int target = code_of(w);
        SDR ctx = ctx_of(p2, p1);
        int rank;
        if (use_readout) {
          rank = ro.rank(ctx, target);
        } else {
          auto ranked = mat.scores(ctx);
          size_t take = std::min<size_t>(256, ranked.size());
          std::vector<float> sc(word_codes.size(), 0.0f);
          std::vector<int> hits(word_codes.size(), 0);
          for (size_t r = 0; r < take; ++r) {
            double norm = ranked[r].second / (bit_freq[ranked[r].first] + 1.0);
            for (int v : bit_to_words[ranked[r].first]) {
              sc[static_cast<size_t>(v)] += static_cast<float>(norm);
              hits[static_cast<size_t>(v)]++;
            }
          }
          float tv = sc[static_cast<size_t>(target)];
          rank = 0;
          for (size_t v = 0; v < sc.size(); ++v) {
            if (static_cast<int>(v) == target || !hits[v]) continue;
            if (sc[v] > tv) ++rank;
          }
        }
        ev.add(rank, target == UNK);
        p2 = p1;
        p1 = target;
      }
    }
    return ev;
  };

  evaluate(true).report("delta-readout");
  evaluate(false).report("hebbian-matrix");

  // context control: fixed context
  {
    Eval ev;
    for (const auto& s : test) {
      for (const auto& w : s) {
        int target = code_of(w);
        SDR ctx = ctx_of(0, 0);
        ev.add(ro.rank(ctx, target), target == UNK);
      }
    }
    ev.report("control-fixed-ctx");
  }

  {
    int top = 0;
    size_t n = 0, hit = 0;
    for (const auto& s : test) {
      for (const auto& w : s) {
        ++n;
        hit += code_of(w) == top;
      }
    }
    printf("unigram-baseline   top1=%.4f (predicting '%s')\n", double(hit) / n,
           vocab_list[0].c_str());
  }

  printf("peak_rss=%zuMB\n", PeakRssMb());
  return 0;
}