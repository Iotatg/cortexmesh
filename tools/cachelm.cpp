// Hashed n-gram memory: a non-saturating alternative to the dense Hebbian matrix.
//
//   cachelm <corpus> <V> <max_tokens> <eval_tokens> <bits> <max_order> <lambda> <decay>
//
// Scores every next-token prediction from hashed contexts of order 2..max_order, with a backoff
// weight lambda^(k_min - k) so that longer contexts dominate when they hit. Reports the same
// top-1 / top-3 / top-10 / MRR as tools/streamlm.cpp so the two are directly comparable, plus the
// cache load and the number of live contexts.
//
// The word ids here are memory addresses, not a representation of meaning: a context is hashed to
// 64 bits and the table stores the id that followed. Meaning still lives in bm::SDR.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "bm/ngram_cache.hpp"

namespace {

std::vector<std::string> Tokenize(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'') {
      cur.push_back(c);
    } else {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

struct Stats {
  long n = 0;
  long top1 = 0, top3 = 0, top10 = 0;
  double mrr = 0.0;
  long oov = 0;

  void add(long rank, bool out_of_vocab) {
    ++n;
    if (out_of_vocab) ++oov;
    if (rank <= 1) ++top1;
    if (rank <= 3) ++top3;
    if (rank <= 10) ++top10;
    if (rank > 0 && rank <= 100) mrr += 1.0 / static_cast<double>(rank);
  }

  void report(const char* name) const {
    if (n == 0) {
      printf("%-16s n=      0\n", name);
      return;
    }
    printf("%-16s n=%8ld top1=%.4f top3=%.4f top10=%.4f mrr=%.4f oov=%.4f\n", name, n,
           static_cast<double>(top1) / n, static_cast<double>(top3) / n,
           static_cast<double>(top10) / n, mrr / static_cast<double>(n),
           static_cast<double>(oov) / n);
  }
};

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "data/wikitext2.txt";
  const size_t V = argc > 2 ? std::stoul(argv[2]) : 10000;
  const long max_tokens = argc > 3 ? std::stol(argv[3]) : 3000000;
  const long eval_tokens = argc > 4 ? std::stol(argv[4]) : 0;
  const int bits = argc > 5 ? std::stoi(argv[5]) : 21;
  const int max_order = argc > 6 ? std::stoi(argv[6]) : 4;
  const double lambda = argc > 7 ? std::stod(argv[7]) : 0.4;
  const double decay = argc > 8 ? std::stod(argv[8]) : 1.0;

  const auto t0 = std::chrono::steady_clock::now();

  // pass 1: vocabulary by frequency
  std::ifstream in(path);
  if (!in) {
    printf("cannot open %s\n", path.c_str());
    return 1;
  }
  std::unordered_map<std::string, long> freq;
  freq.reserve(300000);
  std::string line;
  long n_sents = 0, n_tokens = 0;
  std::vector<std::string> tokens;
  tokens.reserve(1 << 20);
  while (std::getline(in, line)) {
    auto toks = Tokenize(line);
    if (toks.size() < 3) continue;
    ++n_sents;
    for (const auto& t : toks) {
      freq[t]++;
      ++n_tokens;
    }
    if (n_tokens >= max_tokens + eval_tokens) {
      // keep only what we will read, but we still need the counts to be representative
    }
  }
  printf("corpus=%s sentences=%ld tokens=%ld types=%zu\n", path.c_str(), n_sents, n_tokens,
         freq.size());

  std::vector<std::pair<std::string, long>> by_freq(freq.begin(), freq.end());
  std::sort(by_freq.begin(), by_freq.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::unordered_map<std::string, int> index;
  index.reserve(std::min(V, by_freq.size()) * 2 + 8);
  for (size_t i = 0; i < std::min(V, by_freq.size()); ++i) index[by_freq[i].first] = static_cast<int>(i);
  const int UNK = static_cast<int>(std::min(V, by_freq.size()));

  auto code_of = [&](const std::string& w) {
    auto it = index.find(w);
    return it == index.end() ? UNK : it->second;
  };

  bm::NgramCache cache(static_cast<size_t>(bits));

  // pass 2: stream the corpus once, online
  std::ifstream in2(path);
  std::vector<uint32_t> hist;
  hist.reserve(8);
  Stats ev, tr;
  long seen = 0;
  bool learning = true;
  std::vector<std::pair<double, int>> cand;
  std::vector<int> touched;
  while (std::getline(in2, line)) {
    auto toks = Tokenize(line);
    if (toks.size() < 3) continue;
    for (const auto& tok : toks) {
      if (seen >= max_tokens + eval_tokens) break;
      const int target = code_of(tok);

      // score: longest matching context wins, shorter ones back off by lambda per step
      cand.clear();
      uint16_t cval[bm::NgramCache::kMaxCont];
      uint8_t ccnt[bm::NgramCache::kMaxCont];
      for (int k = 2; k <= max_order && static_cast<int>(hist.size()) >= k; ++k) {
        uint32_t ids[8];
        for (int i = 0; i < k; ++i) ids[i] = hist[hist.size() - k + i];
        const uint64_t key = bm::NgramCache::hash_context(ids, static_cast<size_t>(k));
        const int n = cache.get(key, cval, ccnt);
        for (int c = 0; c < n; ++c) {
          const double w = std::pow(lambda, static_cast<double>(max_order - k)) *
                           std::log1p(static_cast<double>(ccnt[c]));
          cand.emplace_back(w, static_cast<int>(cval[c]));
        }
      }

      long rank = 1 << 20;
      if (!cand.empty()) {
        std::sort(cand.begin(), cand.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        // rank of the target among the distinct continuations the cache can name
        rank = 1 << 20;
        for (size_t i = 0; i < cand.size(); ++i) {
          if (cand[i].second == target) {
            rank = static_cast<long>(i) + 1;
            break;
          }
        }
      }
      if (target == UNK) rank = 1 << 20;
      if (learning) {
        tr.add(rank, target == UNK);
      } else {
        ev.add(rank, target == UNK);
      }

      if (learning) {
        // reinforce every context that ends at this token, longest first
        for (int k = 2; k <= max_order && static_cast<int>(hist.size()) >= k; ++k) {
          uint32_t ids[8];
          for (int i = 0; i < k; ++i) ids[i] = hist[hist.size() - k + i];
          const uint64_t key = bm::NgramCache::hash_context(ids, static_cast<size_t>(k));
          cache.put(key, static_cast<uint16_t>(target));
        }
      }

      hist.push_back(static_cast<uint32_t>(target));
      ++seen;
      if (decay < 1.0 && seen % 1000000 == 0) cache.sweep(1.0 - decay);
      if (cache.load() > 0.9) cache.sweep(0.15);
      if (eval_tokens > 0 && seen == max_tokens) {
        learning = false;
        printf("--- eval split begins at token %ld ---\n", seen);
      }
    }
    if (seen >= max_tokens + eval_tokens) break;
  }

  tr.report("train");
  if (eval_tokens > 0) ev.report("eval");
  printf("cache_slots=%zu live=%zu load=%.3f bytes=%.1fMB\n", cache.capacity(), cache.used(),
         cache.load(), static_cast<double>(cache.bytes()) / (1024.0 * 1024.0));
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("seconds=%.1f throughput=%.0f tokens/s\n", secs, static_cast<double>(seen) / secs);
  return 0;
}