// Streaming language model: one pass, EMA Hebbian, bounded memory, online eval.
// Research basis: docs/23_efficient_language_research.md
//   - normalised EMA Hebbian (arXiv:2503.02057)
//   - no stored pairs, no epochs (arXiv:2303.11934, 2111.05498)
//   - sparse decode through an inverted index over the strongest predicted bits
#include <algorithm>
#include <cctype>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bm/encoder.hpp"
#include "bm/cleanup.hpp"
#include "bm/ngram_cache.hpp"
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
  SDR o(s.size(), 0);
  for (uint32_t i : s.indices()) o.set(static_cast<uint32_t>((i + r) % n));
  return o;
}

size_t PeakRssMb() {
  std::ifstream st("/proc/self/status");
  std::string line;
  while (std::getline(st, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      return static_cast<size_t>(std::stol(line.substr(6)) / 1024);
    }
  }
  return 0;
}

struct Stats {
  long n = 0, h1 = 0, h3 = 0, h10 = 0, unk = 0, inv_n = 0, inv_h1 = 0, inv_h10 = 0;
  double mrr = 0;
  void add(int rank, bool out_of_vocab = false) {
    ++n;
    unk += out_of_vocab ? 1 : 0;
    if (!out_of_vocab) {
      ++inv_n;
      inv_h1 += rank == 0;
      inv_h10 += rank < 10;
    }
    h1 += rank == 0;
    h3 += rank < 3;
    h10 += rank < 10;
    mrr += 1.0 / (1.0 + std::max(0, rank));
  }
  void report(const char* tag) const {
    printf("%-16s n=%7ld top1=%.4f top3=%.4f top10=%.4f mrr=%.4f oov=%.4f inv_top1=%.4f "
           "inv_top10=%.4f\n",
           tag, n, n ? double(h1) / n : 0, n ? double(h3) / n : 0, n ? double(h10) / n : 0,
           n ? mrr / n : 0, n ? double(unk) / n : 0, inv_n ? double(inv_h1) / inv_n : 0,
           inv_n ? double(inv_h10) / inv_n : 0);
  }
};

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "data/wikitext2.txt";
  const size_t V = argc > 2 ? std::stoul(argv[2]) : 5000;
  const float xi = argc > 3 ? std::stof(argv[3]) : 0.01f;
  const bool normalize = argc > 4 ? std::stoi(argv[4]) != 0 : true;
  const long max_tokens = argc > 5 ? std::stol(argv[5]) : -1;
  const bool gate = argc > 6 ? std::stoi(argv[6]) != 0 : false;
  const std::string eval_arg = argc > 7 ? argv[7] : "none";
  const std::string eval_path = (eval_arg == "none" || eval_arg == "0" || eval_arg == "-") ? "" : eval_arg;
  const int subword = argc > 8 ? std::stoi(argv[8]) : 0;
  const bool ctx2 = argc > 9 ? std::stoi(argv[9]) != 0 : false;
  const int generate = argc > 10 ? std::stoi(argv[10]) : 0;
  (void)0;
  const std::string seed = argc > 11 ? argv[11] : "the";
  const float temp = argc > 12 ? std::stof(argv[12]) : 1.0f;
  const int npos = argc > 13 ? std::stoi(argv[13]) : 0;
  // decode: 0 = linear bit-score over vocabulary (default), 1 = cleanup memory
  const int decode = argc > 14 ? std::stoi(argv[14]) : 0;
  // how strongly the cleanup-snapped code re-ranks the linear scores
  const float blend = argc > 15 ? std::stof(argv[15]) : 0.5f;
  // number of passes over the corpus (online learning improves with repetition)
  const int epochs = argc > 16 ? std::stoi(argv[16]) : 1;
  // 0 = plain (1-w) Hebb, 1 = importance-scaled (SI/MAS analogue), 2 = mean-subtracted + decay
  const int rule = argc > 17 ? std::stoi(argv[17]) : 0;
  const float lam = argc > 18 ? std::stof(argv[18]) : 10.0f;
  const float wcap = argc > 19 ? std::stof(argv[19]) : 0.6f;
  // read-out competition: 0 = plain sum, 1 = mean-subtract candidate bits, 2 = mean-subtract + drop negatives
  const int comp = argc > 20 ? std::stoi(argv[20]) : 0;
  // hashed n-gram cache interpolation (D47): bits=0 disables it
  const int cache_bits = argc > 21 ? std::stoi(argv[21]) : 0;
  const int cache_order = argc > 22 ? std::stoi(argv[22]) : 3;
  const float cache_beta = argc > 23 ? std::stof(argv[23]) : 0.5f;
  const float cache_lambda = argc > 24 ? std::stof(argv[24]) : 0.4f;
  const size_t N = 512, K = 24;

  // pass 1: count tokens, no storage of pairs
  std::ifstream in(path);
  if (!in) {
    printf("cannot open %s\n", path.c_str());
    return 1;
  }
  std::unordered_map<std::string, long> freq;
  freq.reserve(200000);
  std::string line;
  long n_tokens = 0;
  long n_sents = 0;
  while (std::getline(in, line)) {
    auto toks = Tokenize(line);
    if (toks.size() < 3) continue;
    for (const auto& t : toks) freq[t]++;
    n_tokens += static_cast<long>(toks.size());
    ++n_sents;
  }
  printf("corpus=%s sentences=%ld tokens=%ld types=%zu\n", path.c_str(), n_sents, n_tokens,
         freq.size());

  std::vector<std::pair<std::string, long>> by_freq(freq.begin(), freq.end());
  std::sort(by_freq.begin(), by_freq.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::vector<std::string> vocab;
  std::unordered_map<std::string, int> index;
  for (size_t i = 0; i < std::min(V, by_freq.size()); ++i) {
    index[by_freq[i].first] = static_cast<int>(vocab.size());
    vocab.push_back(by_freq[i].first);
  }
  const int UNK = static_cast<int>(vocab.size());
  vocab.push_back("<unk>");

  Encoder enc(N, K, 5);
  auto subword_code = [&](const std::string& w) {
    std::string p = "<" + w + ">";
    std::vector<SDR> parts;
    for (size_t i = 0; i + 3 <= p.size(); ++i) parts.push_back(enc.symbol("ng:" + p.substr(i, 3)));
    if (parts.empty()) parts.push_back(enc.symbol("w:" + w));
    return enc.bundle(parts);
  };
  auto hybrid_code = [&](const std::string& w) {
    SDR base = subword_code(w);
    SDR id = enc.symbol("id:" + w);
    SDR out(N, 0);
    auto b = base.indices();
    auto d = id.indices();
    size_t half = N / 2;
    for (size_t i = 0; i < b.size() && i < half; ++i) out.set(b[i]);
    for (size_t i = 0; i < d.size(); ++i) out.set(static_cast<uint32_t>(half + d[i] % half));
    return out;
  };
  std::vector<SDR> codes;
  codes.reserve(vocab.size());
  for (const auto& w : vocab) {
    codes.push_back(subword == 1 ? subword_code(w)
                                 : (subword == 2 ? hybrid_code(w) : enc.symbol(w)));
  }
  auto code_of = [&](const std::string& w) {
    auto it = index.find(w);
    return it == index.end() ? UNK : it->second;
  };

  // age-of-acquisition tracking (docs/36, docs/37): a child's learning curve is measured
  // per word, not per corpus. We record the token index at which each word is first
  // predicted inside the top 5, which is the closest cheap proxy for "acquired".
  const bool want_aoa = argc > 21 ? std::stoi(argv[21]) != 0 : false;
  std::vector<long> first_hit(vocab.size(), -1);
  std::vector<long> vocab_freq(vocab.size(), 0);
  for (size_t i = 0; i < vocab.size() && i < by_freq.size(); ++i) vocab_freq[i] = by_freq[i].second;

  // weights: w[i * N + j], plus per-bit frequency for normalisation
  const size_t ctx_size = static_cast<size_t>(npos > 0 ? npos : 1) * N;
  std::vector<float> w(ctx_size * N, 0.0f);
  std::vector<float> omega(ctx_size * N, 0.0f);
  std::vector<int> hist(npos > 0 ? npos : 1, UNK);
  bm::NgramCache ncache(static_cast<size_t>(cache_bits > 0 ? cache_bits : 1));
  bm::NgramCache* ncm = cache_bits > 0 ? &ncache : nullptr;
  auto build_ctx = [&]() {
    SDR ctx(ctx_size, 0);
    for (size_t b = 0; b < hist.size(); ++b) {
      for (uint32_t i : codes[static_cast<size_t>(hist[b])].indices())
        ctx.set(static_cast<uint32_t>(b * N + i));
    }
    return ctx;
  };
  auto push_hist = [&](int w) {
    for (size_t b = hist.size() - 1; b > 0; --b) hist[b] = hist[b - 1];
    hist[0] = w;
  };
  std::vector<float> freq_bit(N, 0.0f);
  std::vector<std::vector<int>> bit_to_words(N);
  for (size_t v = 0; v < codes.size(); ++v) {
    if (static_cast<int>(v) == UNK) continue;  // rank over known words only
    for (uint32_t i : codes[v].indices()) bit_to_words[i].push_back(static_cast<int>(v));
  }

  Stats all, first, last;
  int prev = UNK;
  int prev2 = UNK;
  const long eval_from = n_tokens / 2;
  long seen = 0;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<float> wscore(vocab.size(), 0.0f);
  std::vector<unsigned char> wmark(vocab.size(), 0);
  std::vector<int> touched;
  std::vector<std::pair<float, uint32_t>> bits(N, {0.0f, 0u});

  const size_t code_k = codes.empty() ? 0 : codes[0].count();
  const CleanupMemory* cm = nullptr;
  int decode_mode = 0;

  auto predict = [&](const SDR& ctx) {
    std::fill(bits.begin(), bits.end(), std::make_pair(0.0f, 0u));
    for (uint32_t i : ctx.indices()) {
      const float* row = w.data() + static_cast<size_t>(i) * N;
      for (size_t j = 0; j < N; ++j) bits[j].first += row[j];
    }
    for (size_t j = 0; j < N; ++j) bits[j].second = static_cast<uint32_t>(j);
    size_t take = std::min<size_t>(64, bits.size());
    std::partial_sort(bits.begin(), bits.begin() + take, bits.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    touched.clear();
    float bit_base = 0.0f;
    if (comp > 0) {
      // read-out competition (D37 follow-up): the 64 candidate bits are ranked
      // against each other rather than summed absolutely. Subtracting the mean
      // turns the top-k into a winner-take-most step, so a bit that only looks
      // strong because the whole matrix is uniformly high no longer votes.
      for (size_t r = 0; r < take; ++r) bit_base += bits[r].first;
      bit_base /= static_cast<float>(take);
    }
    for (size_t r = 0; r < take; ++r) {
      float sc = bits[r].first - bit_base;
      if (comp == 2 && sc < 0.0f) continue;
      sc /= (freq_bit[bits[r].second] + 0.05f);
      for (int v : bit_to_words[bits[r].second]) {
        if (!wmark[static_cast<size_t>(v)]) {
          wmark[static_cast<size_t>(v)] = 1;
          touched.push_back(v);
        }
        wscore[static_cast<size_t>(v)] += sc;
          }

          // rule 6: importance-scaled Hebb + synaptic homeostasis downscaling (SHY).
          // Measured biology: after sleep the axon-spine interface of the weakest 80% of
          // synapses shrinks ~18% relative to wake, proportional to synapse size, while the
          // largest synapses are spared (de Vivo et al. 2017, Science). That is exactly the
          // term a positive-only accumulator lacks: it removes mass from the tail instead of
          // capping the head, so ratios inside a column survive.
          if (rule == 6 && seen % 400000 == 0) {
            float lo = w[0], hi = w[0];
            for (float v : w) {
              if (v < lo) lo = v;
              if (v > hi) hi = v;
            }
            float thr = lo;
            if (hi > lo) {
              const int B = 256;
              std::vector<int> hist(B, 0);
              const float span = hi - lo;
              for (float v : w) {
                int b = static_cast<int>((v - lo) / span * (B - 1));
                hist[b]++;
              }
              const long want = static_cast<long>(0.8 * static_cast<double>(w.size()));
              long acc = 0;
              for (int b = 0; b < B; ++b) {
                acc += hist[b];
                if (acc >= want) {
                  thr = lo + span * static_cast<float>(b) / static_cast<float>(B - 1);
                  break;
                }
              }
            }
            const float s = 1.0f - 0.18f;
            for (float& v : w) {
              if (v < thr) v *= s;
            }
          }
        }
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(touched.size() + 16);
    float best_score = 0.0f;
    for (int v : touched) {
      const float sc = wscore[static_cast<size_t>(v)];
      ranked.emplace_back(sc, v);
      if (sc > best_score) best_score = sc;
    }
    for (int v : touched) {
      wscore[static_cast<size_t>(v)] = 0.0f;
      wmark[static_cast<size_t>(v)] = 0;
    }

    if (decode_mode && cm != nullptr && code_k > 0 && !ranked.empty()) {
      SDR pred(N, 0);
      size_t take_k = std::min<size_t>(code_k, take);
      for (size_t r = 0; r < take_k; ++r) pred.set(bits[r].second);
      SDR snapped = cm->clean(pred);
      const float mx = ranked[0].first > 0 ? ranked[0].first : 1.0f;
      size_t n = std::min<size_t>(ranked.size(), 64);
      for (size_t i = 0; i < n; ++i) {
        float ov = static_cast<float>(snapped.overlap(codes[static_cast<size_t>(ranked[i].second)])) /
                   static_cast<float>(code_k);
        ranked[i].first = (1.0f - blend) * (ranked[i].first / mx) + blend * ov;
      }
      std::sort(ranked.begin(), ranked.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
    }
    if (cache_bits > 0 && ncm != nullptr) {
      // interpolate the dense matrix with the hashed n-gram cache (D47 follow-up).
      // The two fail differently: the matrix generalises but saturates, the cache is exact
      // but only where it has seen the context. Both are normalised to [0,1] before blending
      // so neither scale dominates the other.
      uint16_t cval[bm::NgramCache::kMaxCont];
      uint8_t ccnt[bm::NgramCache::kMaxCont];
      std::vector<std::pair<float, int>> hc;
      for (int k = 2; k <= cache_order && static_cast<int>(hist.size()) >= k; ++k) {
        uint32_t ids[8];
        for (int i = 0; i < k; ++i) ids[i] = hist[hist.size() - k + i];
        const uint64_t key = bm::NgramCache::hash_context(ids, static_cast<size_t>(k));
        const int n = ncm->get(key, cval, ccnt);
        for (int c = 0; c < n; ++c) {
          const float s = std::pow(static_cast<float>(cache_lambda),
                                   static_cast<float>(cache_order - k)) *
                          std::log1p(static_cast<float>(ccnt[c]));
          hc.emplace_back(s, static_cast<int>(cval[c]));
        }
      }
      if (!hc.empty()) {
        float mx = 0.0f;
        for (const auto& p : hc) mx = std::max(mx, p.first);
        if (mx > 0.0f) {
          // scale the cache to the matrix's own top score, so cache_beta means the same
          // thing regardless of how large the bit sums happen to be
          const float scale = best_score > 0 ? best_score : 1.0f;
          for (const auto& p : hc) {
            const int v = p.second;
            const float add = cache_beta * (p.first / mx) * scale;
            auto it = std::find_if(ranked.begin(), ranked.end(),
                                   [v](const auto& q) { return q.second == v; });
            if (it == ranked.end()) ranked.emplace_back(add, v);
            else it->first += add;
          }
          // no sort here: the caller ranks by counting how many candidates beat the target,
          // which is order-independent. Sorting a 20k-element vector per token is what made
          // the first version of this 2.3x slower than the matrix alone.
        }
      }
    }
    return ranked;
  };

  auto run_pass = [&](const std::string& file, bool learning, bool split, Stats* st,
                      long budget = 0) {
    std::ifstream in(file);
    if (!in) {
      printf("cannot open %s\n", file.c_str());
      return;
    }
    [[maybe_unused]] int prev_local = prev;
    long local_seen = 0;
    std::string line;
    while (std::getline(in, line)) {
      auto s = Tokenize(line);
      if (s.size() < 3) continue;
      for (const auto& tok : s) {
        const int target = code_of(tok);
        SDR ctx = npos > 0 ? build_ctx()
                           : (ctx2 ? codes[static_cast<size_t>(prev)] |
                                       Rotate(codes[static_cast<size_t>(prev2)], N / 2)
                                   : codes[static_cast<size_t>(prev)]);
        const SDR& tgt = codes[static_cast<size_t>(target)];

        // predict before learning
        auto ranked = predict(ctx);
        int rank = 0;
        float tv = 0.0f;
        for (const auto& pr : ranked) {
          if (pr.second == target) {
            tv = pr.first;
            break;
          }
        }
        for (const auto& pr : ranked) {
          if (pr.second != target && pr.first > tv) ++rank;
        }
        if (target == UNK) rank = 1 << 20;
        if (want_aoa && target != UNK && rank <= 5 && first_hit[target] < 0) first_hit[target] = seen;
        (st ? *st : all).add(rank, target == UNK);
        if (split) {
          if (seen < eval_from) first.add(rank);
          if (seen >= n_tokens - eval_from) last.add(rank);
        }

        if (learning) {
          const float mx = 1.0f / static_cast<float>(ctx.count());
          const float my = 1.0f / static_cast<float>(tgt.count());
          for (uint32_t i : ctx.indices()) {
            float* row = w.data() + static_cast<size_t>(i) * N;
            for (uint32_t j : tgt.indices()) {
              float g = gate ? (0.2f + 0.8f * row[j]) : 1.0f;
              if (rule == 1 || rule == 6) {
                float& om = omega[static_cast<size_t>(i) * N + j];
                const float dw = (xi * g / (1.0f + lam * om)) * (1.0f - row[j]);
                row[j] += dw;
                om += std::fabs(dw);
              } else if (rule == 3) {
                // importance-scaled Hebb with a hard per-synapse cap (D42 follow-up).
                // Positive-only accumulation has no fixed point below 1, so without a
                // ceiling every synapse eventually saturates and the matrix stops
                // discriminating. The cap keeps a margin between winner and loser.
                float& om = omega[static_cast<size_t>(i) * N + j];
                const float dw = (xi * g / (1.0f + lam * om)) * (1.0f - row[j]);
                row[j] = std::min(row[j] + dw, wcap);
                om += std::fabs(dw);
              } else if (rule == 4) {
                // importance-scaled Hebb with a soft target instead of a hard cap (D43).
                // A hard cap pins every synapse to the same value and destroys the
                // ranking (measured: 75% of synapses at the cap, top-1 fell to 0.027).
                // Here each synapse relaxes towards wcap, and the rate slows with
                // accumulated importance, so the ones that matter arrive first and
                // the rest stay low. Equilibrium is wcap, but it is approached
                // asymptotically, which is what keeps the matrix discriminative.
                float& om = omega[static_cast<size_t>(i) * N + j];
                const float dw = (xi * g / (1.0f + lam * om)) * (1.0f - row[j] / wcap);
                row[j] += dw;
                om += std::fabs(dw);
              } else if (rule == 2) {
                row[j] += xi * g * ((1.0f - mx) * (1.0f - my) - row[j]);
              } else {
                row[j] += xi * g * (1.0f - row[j]);
              }
            }
          }
          for (uint32_t j : tgt.indices()) freq_bit[j] += xi * (1.0f - freq_bit[j]);

          // the cache learns from the same context, but it stores the observed continuation
          // instead of a weight, so it cannot saturate
          if (ncm != nullptr && target != UNK) {
            for (int k = 2; k <= cache_order && static_cast<int>(hist.size()) >= k; ++k) {
              uint32_t ids[8];
              for (int i = 0; i < k; ++i) ids[i] = hist[hist.size() - k + i];
              ncm->put(bm::NgramCache::hash_context(ids, static_cast<size_t>(k)),
                       static_cast<uint16_t>(target));
            }
            if (ncm->load() > 0.9) ncm->sweep(0.15);
          }

          // rule 5: ratio-preserving column renormalisation (D43 follow-up).
          // Hard caps and soft targets both change the ranking, because they
          // change ratios between synapses. Scaling a whole column by one factor
          // does not: the ranking inside a column is untouched, while the absolute
          // scale stops drifting upwards, which is what saturates the matrix.
          if (rule == 5 && seen % 200000 == 0) {
            std::vector<float> colmax(N, 0.0f);
            for (size_t i = 0; i < w.size(); ++i) {
              const float v = w[i];
              if (v > colmax[i % N]) colmax[i % N] = v;
            }
            for (uint32_t c = 0; c < N; ++c) {
              if (colmax[c] <= wcap) continue;
              const float s = wcap / colmax[c];
              for (size_t i = c; i < w.size(); ++i += N) w[i] *= s;
            }
          }
        }

        prev2 = prev;
        prev = target;
        if (npos > 0) push_hist(target);
        ++seen;
        ++local_seen;
        if (budget > 0 && local_seen >= budget) return;
        if (max_tokens > 0 && local_seen >= max_tokens) return;
        if (seen % 200000 == 0) {
          auto now = std::chrono::steady_clock::now();
          printf("  seen=%ld elapsed=%.1fs rss=%zuMB\n", seen,
                 std::chrono::duration<double>(now - t0).count(), PeakRssMb());
        }
      }
      if (max_tokens > 0 && local_seen >= max_tokens) return;
    }
  };

  for (int ep = 0; ep < epochs; ++ep) {
    all = Stats{};
    first = Stats{};
    last = Stats{};
    run_pass(path, true, true, nullptr);
    char tag[64];
    snprintf(tag, sizeof(tag), "epoch%d-second-half", ep + 1);
    all.report(tag);
    printf("epoch%d_seconds=%.1f\n", ep + 1,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (ep + 1 < epochs) {
      seen = 0;              // restart the clock for the next pass
      prev = UNK;
      prev2 = UNK;
      hist.assign(hist.size(), UNK);
    }
  }
  auto t1 = std::chrono::steady_clock::now();

  all.report("train-all");
  first.report("train-first-half");
  last.report("train-second-half");
  printf("train_seconds=%.2f peak_rss=%zuMB xi=%.3f normalize=%d gate=%d subword=%d pos=%d\n",
         std::chrono::duration<double>(t1 - t0).count(), PeakRssMb(), xi, int(normalize),
         int(gate), subword, npos);

  if (decode >= 1 && code_k > 0) {
    CleanupMemory local_cm(N, codes.size() - 1);
    for (size_t v = 0; v < codes.size(); ++v)
      if (static_cast<int>(v) != UNK) local_cm.add(codes[v]);
    cm = &local_cm;
    decode_mode = (decode == 1) ? 1 : 0;
    Stats cd;
    auto c0 = std::chrono::steady_clock::now();
    run_pass(path, false, false, &cd, 50000);
    cd.report(decode == 1 ? "cleanup-decode" : "baseline-decode");
    printf("cleanup_decode_seconds=%.2f codes=%zu ram=%.1fMB k=%zu\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count(),
           local_cm.size(), double(local_cm.bytes()) / (1024.0 * 1024.0), code_k);
  }

  // diagnostic: is the weight matrix saturating? (D41)
  {
    double sum = 0.0, gt9 = 0.0, gt5 = 0.0, nz = 0.0, hit = 0.0;
    for (float v : w) {
      sum += v;
      if (v > 0.9f) ++gt9;
      if (v > 0.5f) ++gt5;
      if (v > 0.01f) ++nz;
      if (v >= wcap - 1e-6f) ++hit;
    }
    const double n = static_cast<double>(w.size());
    printf("diag,mean=%.4f,frac_gt_0.9=%.4f,frac_gt_0.5=%.4f,frac_gt_0.01=%.4f,frac_gt_cap=%.4f\n", sum / n,
           gt9 / n, gt5 / n, nz / n, hit / n);
  }

  if (want_aoa) {
    // Report the acquisition curve: for each frequency decile, the median token index at
    // which its words are first predicted top-5, plus the rank correlation between corpus
    // frequency and AoA. Child data predicts a strong negative correlation (frequent words
    // acquired early); a system that only tracks n-grams predicts a weak one.
    const int D = 10;
    std::vector<std::vector<long>> dec(D);
    std::vector<double> lf, la;
    for (size_t v = 0; v < vocab.size(); ++v) {
      if (v == static_cast<size_t>(UNK)) continue;
      if (first_hit[v] < 0) continue;
      const long f = vocab_freq[v];
      if (f <= 0) continue;
      const double rank = static_cast<double>(v) / static_cast<double>(vocab.size());
      dec[std::min<int>(D - 1, static_cast<int>(rank * D))].push_back(first_hit[v]);
      lf.push_back(std::log(static_cast<double>(f)));
      la.push_back(std::log(static_cast<double>(first_hit[v]) + 1.0));
    }
    for (int d = 0; d < D; ++d) {
      auto& xs = dec[d];
      long med = -1;
      if (!xs.empty()) {
        std::sort(xs.begin(), xs.end());
        med = xs[xs.size() / 2];
      }
      printf("aoa_decile %d n=%zu median_first_hit=%ld\n", d, xs.size(), med);
    }
    double mf = 0, ma = 0;
    for (size_t i = 0; i < lf.size(); ++i) {
      mf += lf[i];
      ma += la[i];
    }
    mf /= std::max<size_t>(1, lf.size());
    ma /= std::max<size_t>(1, lf.size());
    double cov = 0, vf = 0, va = 0;
    for (size_t i = 0; i < lf.size(); ++i) {
      cov += (lf[i] - mf) * (la[i] - ma);
      vf += (lf[i] - mf) * (lf[i] - mf);
      va += (la[i] - ma) * (la[i] - ma);
    }
    const double r = (vf > 0 && va > 0) ? cov / std::sqrt(vf * va) : 0.0;
    printf("aoa_pearson_logfreq_logaoa=%.4f n=%zu\n", r, lf.size());
  }

  if (eval_path.size()) {
    Stats ev;
    auto e0 = std::chrono::steady_clock::now();
    run_pass(eval_path, false, false, &ev);
    auto e1 = std::chrono::steady_clock::now();
    ev.report("transfer");
    printf("transfer_seconds=%.2f\n", std::chrono::duration<double>(e1 - e0).count());
  }

  if (generate > 0) {
    std::mt19937 rng(12345);
    int cur = code_of(seed);
    if (cur < 0) cur = UNK;
    std::string sentence = seed;
    for (int g = 0; g < generate; ++g) {
      if (npos > 0) push_hist(cur);
      SDR ctx = npos > 0 ? build_ctx() : codes[static_cast<size_t>(cur)];
      auto ranked = predict(ctx);
      if (ranked.empty()) break;
      std::vector<std::pair<float, int>> probs;
      float mx = ranked.front().first;
      float sum = 0.0f;
      for (const auto& pr : ranked) {
        if (pr.first < mx * 0.25f) continue;  // temperature-like floor
        float e = std::exp((pr.first - mx) / (temp * (std::abs(mx) + 1e-3f)));
        probs.emplace_back(e, pr.second);
        sum += e;
      }
      if (probs.empty()) break;
      std::uniform_real_distribution<float> uni(0.0f, 1.0f);
      float r = uni(rng) * sum, acc = 0.0f;
      int pick = probs.back().second;
      for (const auto& pr : probs) {
        acc += pr.first;
        if (r <= acc) {
          pick = pr.second;
          break;
        }
      }
      sentence += " " + vocab[static_cast<size_t>(pick)];
      cur = pick;
    }
    printf("generate: %s\n", sentence.c_str());
  }

  return 0;
}