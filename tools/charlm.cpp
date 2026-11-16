// Character-level streaming language model: one pass, EMA Hebbian, bounded memory.
// Same mechanism as tools/streamlm.cpp (context SDR -> weight matrix -> sparse readout),
// but the output space is characters, so there is no OOV by construction (D27).
//
// Context encoding: positional concatenation. Each of the k previous characters gets its
// own block of N bits (position p occupies [p*N, (p+1)*N)). This keeps order: the same
// character in two positions activates disjoint bits. OR-binding is not used because it
// loses order (D23/D25).
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "bm/encoder.hpp"
#include "bm/sdr.hpp"

using namespace bm;

namespace {

size_t g_n = 256;                 // bits per character code (runtime, argv[11])
constexpr size_t kK = 8;       // active bits per character (3.1% of N)
size_t g_ctx = 2;                // characters of context (runtime, argv[9])
constexpr size_t kTopBits = 64; // sparse readout width
constexpr bool g_normalize = false;  // bit-frequency normalisation (D24: no gain)
constexpr float kDefaultXi = 0.002f;

bool g_code = false;  // code mode: full printable ASCII, case preserved

bool IsAllowed(unsigned char c) {
  if (g_code) return c >= 0x20 && c <= 0x7e;
  if (c >= 'a' && c <= 'z') return true;
  if (c >= '0' && c <= '9') return true;
  switch (c) {
    case ' ': case '.': case ',': case ';': case ':': case '!': case '?':
    case '\'': case '"': case '(': case ')': case '-':
      return true;
    default:
      return false;
  }
}

unsigned char NormByte(unsigned char c) {
  if (std::isspace(c)) return ' ';
  if (g_code) return c;
  return static_cast<unsigned char>(std::tolower(c));
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::string();
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
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
  long n = 0, h1 = 0, h5 = 0;
  double mrr = 0;
  void add(int rank) {
    ++n;
    h1 += rank == 0;
    h5 += rank < 5;
    mrr += 1.0 / (1.0 + std::max(0, rank));
  }
  double top1() const { return n ? double(h1) / n : 0.0; }
  double top5() const { return n ? double(h5) / n : 0.0; }
  double mrr_score() const { return n ? mrr / n : 0.0; }
};

struct Model {
  size_t n = g_n;
  float xi = kDefaultXi;
  std::vector<SDR> codes;             // alphabet size x N
  std::vector<int> id;                // byte -> alphabet index, -1 if absent
  std::vector<float> w;               // (g_ctx * N) x N
  std::vector<float> freq_bit;        // N, optional normalisation
  std::vector<std::vector<int>> bit_to_char;
  std::vector<float> wscore;
  std::vector<unsigned char> wmark;
  std::vector<int> touched;
  // Split accumulator/indices so the inner product is a contiguous float loop (vectorises).
  std::vector<float> acc;
  std::vector<std::pair<float, uint32_t>> order;

  void init(size_t alphabet) {
    w.assign(g_ctx * n * n, 0.0f);
    freq_bit.assign(n, 0.0f);
    bit_to_char.assign(n, {});
    for (size_t c = 0; c < alphabet; ++c) {
      for (uint32_t b : codes[c].indices()) bit_to_char[b].push_back(static_cast<int>(c));
    }
    wscore.assign(alphabet, 0.0f);
    wmark.assign(alphabet, 0);
    acc.assign(n, 0.0f);
    order.resize(n);
  }

  std::vector<std::pair<float, int>> Predict(const SDR& ctx) {
    std::fill(acc.begin(), acc.end(), 0.0f);
    for (uint32_t i : ctx.indices()) {
      const float* row = w.data() + static_cast<size_t>(i) * n;
      for (size_t j = 0; j < n; ++j) acc[j] += row[j];
    }
    for (size_t j = 0; j < n; ++j) order[j] = {acc[j], static_cast<uint32_t>(j)};
    const size_t take = std::min(kTopBits, order.size());
    std::partial_sort(order.begin(), order.begin() + take, order.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    touched.clear();
    for (size_t r = 0; r < take; ++r) {
      const uint32_t bit = order[r].second;
      float sc = order[r].first;
      if (g_normalize) sc = sc / (freq_bit[bit] + 0.05f);
      for (int c : bit_to_char[bit]) {
        if (!wmark[static_cast<size_t>(c)]) {
          wmark[static_cast<size_t>(c)] = 1;
          touched.push_back(c);
        }
        wscore[static_cast<size_t>(c)] += sc;
      }
    }
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(touched.size());
    for (int c : touched) {
      ranked.emplace_back(wscore[static_cast<size_t>(c)], c);
      wscore[static_cast<size_t>(c)] = 0.0f;
      wmark[static_cast<size_t>(c)] = 0;
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    return ranked;
  }

  void Learn(const SDR& ctx, const SDR& tgt) {
    const auto tidx = tgt.indices();
    for (uint32_t i : ctx.indices()) {
      float* row = w.data() + static_cast<size_t>(i) * n;
      for (uint32_t j : tidx) row[j] += xi * (1.0f - row[j]);
    }
    if (g_normalize) {
      for (uint32_t j : tidx) freq_bit[j] += xi * (1.0f - freq_bit[j]);
    }
  }
};

int RankOf(const std::vector<std::pair<float, int>>& ranked, int target) {
  float tv = 0.0f;
  bool found = false;
  for (const auto& pr : ranked) {
    if (pr.second == target) {
      tv = pr.first;
      found = true;
      break;
    }
  }
  if (!found) {
    // No evidence for the target: rank it last, but never call that a hit.
    return ranked.empty() ? 1 : static_cast<int>(ranked.size());
  }
  int rank = 0;
  for (const auto& pr : ranked) {
    if (pr.second != target && pr.first > tv) ++rank;
  }
  return rank;
}

struct PassOut {
  Stats all, first, second;
  long n_prev = 0, prev_hit = 0, uni_hit = 0;
};

PassOut RunPass(const std::string& text, Model& m, bool learn, int uni_target, bool split,
                long max_chars, bool progress) {
  PassOut out;
  const long limit = max_chars > 0 ? std::min<long>(max_chars, static_cast<long>(text.size()))
                                   : static_cast<long>(text.size());
  const long split_at = limit / 2;
  const int space_id = m.id[static_cast<unsigned char>(' ')] >= 0
                           ? m.id[static_cast<unsigned char>(' ')]
                           : 0;
  std::vector<int> hist(g_ctx, space_id);
  auto t0 = std::chrono::steady_clock::now();

  for (long t = 0; t < limit; ++t) {
    const int target = m.id[static_cast<unsigned char>(text[static_cast<size_t>(t)])];
    if (target < 0) continue;

    SDR ctx(g_ctx * m.n, 0);
    for (size_t p = 0; p < g_ctx; ++p) {
      const SDR& code = m.codes[static_cast<size_t>(hist[p])];
      for (uint32_t b : code.indices()) {
        ctx.set(static_cast<uint32_t>(p * m.n + b));
      }
    }

    const auto ranked = m.Predict(ctx);
    const int rank = RankOf(ranked, target);
    out.all.add(rank);
    if (split) {
      if (t < split_at) {
        out.first.add(rank);
      } else {
        out.second.add(rank);
      }
    }
    if (learn) m.Learn(ctx, m.codes[static_cast<size_t>(target)]);

    if (t > 0) {
      ++out.n_prev;
      if (text[static_cast<size_t>(t)] == text[static_cast<size_t>(t - 1)]) ++out.prev_hit;
    }
    if (target == uni_target) ++out.uni_hit;

    hist.erase(hist.begin());
    hist.push_back(target);

    if (progress && (t + 1) % 200000 == 0) {
      auto now = std::chrono::steady_clock::now();
      fprintf(stderr, "  chars=%ld elapsed=%.1fs rss=%zuMB\n", t + 1,
              std::chrono::duration<double>(now - t0).count(), PeakRssMb());
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "data/wikitext2_head.txt";
  const long vocab_unused = argc > 2 ? std::strtol(argv[2], nullptr, 10) : 0;
  (void)vocab_unused;  // character model has no vocabulary cutoff
  float xi = argc > 3 ? std::strtof(argv[3], nullptr) : kDefaultXi;
  if (!(xi > 0.0f && xi <= 1.0f)) xi = kDefaultXi;
  long max_tokens = argc > 4 ? std::strtol(argv[4], nullptr, 10) : -1;
  if (max_tokens == 0) max_tokens = -1;
  const std::string eval_path = argc > 5 ? argv[5] : "";
  const long generate = argc > 6 ? std::strtol(argv[6], nullptr, 10) : 0;
  const std::string seed = argc > 7 ? argv[7] : "the";
  g_code = argc > 8 && std::strtol(argv[8], nullptr, 10) != 0;
  g_n = argc > 11 ? static_cast<size_t>(std::strtol(argv[11], nullptr, 10)) : 256;
  if (g_n < 64 || g_n > 4096) g_n = 256;
  g_ctx = argc > 9 ? static_cast<size_t>(std::strtol(argv[9], nullptr, 10)) : 2;
  if (g_ctx < 1 || g_ctx > 32) g_ctx = 2;

  const std::string raw = ReadFile(path);
  if (raw.empty()) {
    printf("cannot open %s\n", path.c_str());
    return 1;
  }

  // Pass 1: collect the characters that are actually present (lower-cased ASCII letters,
  // digits, a few punctuation marks, whitespace folded to space). Everything else is
  // skipped, so every character the model sees is in the alphabet by construction.
  bool present[256] = {false};
  for (unsigned char c : raw) {
    const unsigned char nc = NormByte(c);
    if (IsAllowed(nc)) present[nc] = true;
  }
  std::vector<unsigned char> alphabet_chars;
  for (int c = 0; c < 256; ++c) {
    if (present[c]) alphabet_chars.push_back(static_cast<unsigned char>(c));
  }
  const size_t A = alphabet_chars.size();
  if (A == 0) {
    printf("no usable characters in %s\n", path.c_str());
    return 1;
  }

  Model m;
  m.id.assign(256, -1);
  m.codes.resize(A);
  Encoder enc(g_n, kK, 5);
  for (size_t i = 0; i < A; ++i) {
    const unsigned char c = alphabet_chars[i];
    m.id[c] = static_cast<int>(i);
    m.codes[i] = enc.symbol(std::string(1, static_cast<char>(c)));
  }
  m.xi = xi;
  m.init(A);

  // Filter the training text to the alphabet.
  std::string text;
  text.reserve(raw.size());
  long skipped = 0;
  for (unsigned char c : raw) {
    const unsigned char nc = NormByte(c);
    if (IsAllowed(nc)) {
      text.push_back(static_cast<char>(nc));
    } else {
      ++skipped;
    }
  }

  int uni_char = 0;
  long uni_best = -1;
  {
    long cnt[256] = {0};
    for (unsigned char c : text) ++cnt[c];
    for (int c = 0; c < 256; ++c) {
      if (cnt[c] > uni_best) {
        uni_best = cnt[c];
        uni_char = c;
      }
    }
  }
  const int uni_target = m.id[static_cast<unsigned char>(uni_char)];

  printf("char,corpus=%s chars=%zu skipped=%ld alphabet=%zu k=%zu xi=%.3f "
         "encoding=positional_concat bits=%zu/%zu top_bits=%zu normalize=%d\n",
         path.c_str(), text.size(), skipped, A, g_ctx, xi, kK, g_n, kTopBits, int(g_normalize));

  auto t0 = std::chrono::steady_clock::now();
  PassOut train = RunPass(text, m, true, uni_target, true, max_tokens, true);
  auto t1 = std::chrono::steady_clock::now();

  printf("char,train_top1=%.4f,top5=%.4f,mrr=%.4f,oov=0.000,n_chars=%ld,alphabet=%zu\n",
         train.all.top1(), train.all.top5(), train.all.mrr_score(), train.all.n, A);
  printf("char,first=top1=%.4f,mrr=%.4f\n", train.first.top1(), train.first.mrr_score());
  printf("char,second=top1=%.4f,mrr=%.4f\n", train.second.top1(), train.second.mrr_score());
  printf("char,train_seconds=%.2f peak_rss=%zuMB\n",
         std::chrono::duration<double>(t1 - t0).count(), PeakRssMb());

  long eval_skipped = 0;
  if (!eval_path.empty()) {
    const std::string raw_eval = ReadFile(eval_path);
    if (raw_eval.empty()) {
      printf("cannot open %s\n", eval_path.c_str());
      return 1;
    }
    std::string eval_text;
    eval_text.reserve(raw_eval.size());
    for (unsigned char c : raw_eval) {
      const unsigned char nc = NormByte(c);
      if (IsAllowed(nc) && m.id[nc] >= 0) {
        eval_text.push_back(static_cast<char>(nc));
      } else {
        ++eval_skipped;
      }
    }
    PassOut ev = RunPass(eval_text, m, false, uni_target, false, -1, false);
    printf("char,top1=%.4f,top5=%.4f,mrr=%.4f,oov=0.000,n_chars=%ld,alphabet=%zu\n",
           ev.all.top1(), ev.all.top5(), ev.all.mrr_score(), ev.all.n, A);
    printf("char,prev_baseline=%.4f\n", ev.n_prev ? double(ev.prev_hit) / ev.n_prev : 0.0);
    printf("char,unigram_baseline=%.4f\n", ev.all.n ? double(ev.uni_hit) / ev.all.n : 0.0);
    if (eval_skipped > 0) {
      printf("char,eval_skipped=%ld\n", eval_skipped);
    }
    // greedy continuation benchmark: how far ahead does the model stay correct?
    const long complete = argc > 10 ? std::strtol(argv[10], nullptr, 10) : 0;
    if (complete > 0 && eval_text.size() > 4 * g_ctx + complete) {
      const int space_id = m.id[static_cast<unsigned char>(' ')] >= 0
                               ? m.id[static_cast<unsigned char>(' ')]
                               : 0;
      long samples = 0, exact = 0, pos_tot = 0, pos_hit = 0;
      double prefix = 0.0;
      for (size_t pos = g_ctx; pos + complete < eval_text.size(); pos += 97) {
        std::vector<int> hist(g_ctx, space_id);
        bool usable = true;
        for (size_t p = 0; p < g_ctx; ++p) {
          int id = m.id[static_cast<unsigned char>(eval_text[pos - g_ctx + p])];
          if (id < 0) { usable = false; break; }
          hist[p] = id;
        }
        if (!usable) continue;
        size_t match = 0;
        for (long k = 0; k < complete; ++k) {
          SDR ctx(g_ctx * m.n, 0);
          for (size_t p = 0; p < g_ctx; ++p) {
            const SDR& code = m.codes[static_cast<size_t>(hist[p])];
            for (uint32_t b : code.indices()) ctx.set(static_cast<uint32_t>(p * m.n + b));
          }
          const auto ranked = m.Predict(ctx);
          int pick = ranked.empty() ? -1 : ranked[0].second;
          const unsigned char want = static_cast<unsigned char>(eval_text[pos + k]);
          ++pos_tot;
          if (pick >= 0 && alphabet_chars[static_cast<size_t>(pick)] == want) ++pos_hit;
          if (pick < 0 || alphabet_chars[static_cast<size_t>(pick)] != want) break;
          ++match;
          hist.erase(hist.begin());
          hist.push_back(pick);
        }
        prefix += double(match);
        exact += (match == static_cast<size_t>(complete));
        ++samples;
      }
      if (samples > 0) {
        printf("char,completion,samples=%ld,exact=%.4f,mean_prefix=%.3f,pos_acc=%.4f\n", samples,
               double(exact) / samples, prefix / samples, double(pos_hit) / pos_tot);
      }
      // beam search over the same model: does search fix the error compounding?
      const long beam = argc > 12 ? std::strtol(argv[12], nullptr, 10) : 0;
      if (beam > 1) {
        struct State {
          std::vector<int> hist;
          double score = 0.0;
          std::string out;
        };
        long bsamples = 0, bexact = 0;
        double bprefix = 0.0;
        for (size_t pos = g_ctx; pos + complete < eval_text.size(); pos += 97) {
          std::vector<int> hist(g_ctx, space_id);
          bool usable = true;
          for (size_t p = 0; p < g_ctx; ++p) {
            int id = m.id[static_cast<unsigned char>(eval_text[pos - g_ctx + p])];
            if (id < 0) { usable = false; break; }
            hist[p] = id;
          }
          if (!usable) continue;
          std::vector<State> beam_states(1);
          beam_states[0].hist = hist;
          for (long k = 0; k < complete; ++k) {
            std::vector<State> next;
            for (const auto& st : beam_states) {
              SDR ctx(g_ctx * m.n, 0);
              for (size_t p = 0; p < g_ctx; ++p) {
                const SDR& code = m.codes[static_cast<size_t>(st.hist[p])];
                for (uint32_t b : code.indices()) ctx.set(static_cast<uint32_t>(p * m.n + b));
              }
              const auto ranked = m.Predict(ctx);
              const size_t take = std::min<size_t>(static_cast<size_t>(beam), ranked.size());
              for (size_t r = 0; r < take; ++r) {
                State ns = st;
                const int pick = ranked[r].second;
                ns.score += ranked[r].first;
                ns.out.push_back(alphabet_chars[static_cast<size_t>(pick)]);
                ns.hist.erase(ns.hist.begin());
                ns.hist.push_back(pick);
                next.push_back(std::move(ns));
              }
            }
            std::sort(next.begin(), next.end(),
                      [](const State& a, const State& b) { return a.score > b.score; });
            if (next.size() > static_cast<size_t>(beam)) next.resize(static_cast<size_t>(beam));
            beam_states.swap(next);
          }
          if (beam_states.empty()) continue;
          const std::string& out = beam_states[0].out;
          const std::string want = eval_text.substr(pos, complete);
          size_t match = 0;
          while (match < out.size() && match < want.size() && out[match] == want[match]) ++match;
          bprefix += double(match);
          bexact += (match == complete);
          ++bsamples;
        }
        if (bsamples > 0) {
          printf("char,beam,samples=%ld,width=%ld,exact=%.4f,mean_prefix=%.3f\n", bsamples, beam,
                 double(bexact) / bsamples, bprefix / bsamples);
        }
      }
    }

  } else {
    printf("char,top1=%.4f,top5=%.4f,mrr=%.4f,oov=0.000,n_chars=%ld,alphabet=%zu\n",
           train.all.top1(), train.all.top5(), train.all.mrr_score(), train.all.n, A);
    printf("char,prev_baseline=%.4f\n",
           train.n_prev ? double(train.prev_hit) / train.n_prev : 0.0);
    printf("char,unigram_baseline=%.4f\n",
           train.all.n ? double(train.uni_hit) / train.all.n : 0.0);
  }

  if (generate > 0) {
    std::mt19937 rng(12345);
    const int space_id = m.id[static_cast<unsigned char>(' ')] >= 0
                             ? m.id[static_cast<unsigned char>(' ')]
                             : 0;
    std::vector<int> hist(g_ctx, space_id);
    std::string out;
    for (unsigned char c : seed) {
      const unsigned char nc = NormByte(c);
      const int id = m.id[nc];
      if (id < 0) continue;
      out.push_back(static_cast<char>(nc));
      hist.erase(hist.begin());
      hist.push_back(id);
    }
    for (long g = 0; g < generate; ++g) {
      SDR ctx(g_ctx * m.n, 0);
      for (size_t p = 0; p < g_ctx; ++p) {
        const SDR& code = m.codes[static_cast<size_t>(hist[p])];
        for (uint32_t b : code.indices()) {
          ctx.set(static_cast<uint32_t>(p * m.n + b));
        }
      }
      const auto ranked = m.Predict(ctx);
      if (ranked.empty()) break;

      double mean = 0.0;
      for (const auto& pr : ranked) mean += pr.first;
      mean /= static_cast<double>(ranked.size());
      double var = 0.0;
      for (const auto& pr : ranked) {
        const double d = pr.first - static_cast<float>(mean);
        var += d * d;
      }
      var /= static_cast<double>(ranked.size());
      const double sd = std::sqrt(var) + 1e-3;
      std::vector<float> probs;
      probs.reserve(ranked.size());
      float mx = -1e30f;
      for (const auto& pr : ranked) {
        const float logit =
            static_cast<float>((pr.first - static_cast<float>(mean)) / (sd * 0.8));
        probs.push_back(logit);
        mx = std::max(mx, logit);
      }
      float sum = 0.0f;
      for (float& p : probs) {
        p = std::exp(p - mx);
        sum += p;
      }
      if (!(sum > 0.0f)) break;
      std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
      const int pick = ranked[dist(rng)].second;
      if (pick < 0 || static_cast<size_t>(pick) >= alphabet_chars.size()) break;
      out.push_back(static_cast<char>(alphabet_chars[static_cast<size_t>(pick)]));
      hist.erase(hist.begin());
      hist.push_back(pick);
    }
    printf("char,generate=%s\n", out.c_str());
  }

  printf("char,peak_rss=%zuMB\n", PeakRssMb());
  return 0;
}