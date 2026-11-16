// Which local learning rule survives many patterns?
//
// The plain rule w += xi * g * (1 - w) drives every co-active synapse to 1 (D41): after 8M
// patterns 96.9% of the matrix is above 0.9 and the linear read-out returns a constant.
// This tool compares four local rules on the same retrieval task:
//
//   hebb       w += xi * (1 - w)                       (current, saturating)
//   cov        w += xi * (x_i - mean_x) * (y_j - mean_y)   (mean subtracted, DC removed)
//   delta      w  = w * (1 - beta) + beta * x_i * y_j  (bounded, gate)
//   capped     w += xi * (1 - w), then w = min(w, cap)  (clip)
//
// and reports recall as a function of the number of stored pairs, plus the weight histogram.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "bm/sdr.hpp"

using namespace bm;

namespace {

struct Mem {
  size_t N;
  float xi = 0.05f;
  float cap = 1.0f;
  int rule = 0;
  std::vector<float> w;  // w[i * N + j]

  Mem(size_t n, int r, float x, float c = 1.0f) : N(n), xi(x), cap(c), rule(r), w(n * n, 0.0f) {}

  void learn(const SDR& in, const SDR& out) {
    const size_t k = in.count();
    if (k == 0) return;
    const float mx = 1.0f / static_cast<float>(k);
    const size_t ko = out.count();
    const float my = ko ? 1.0f / static_cast<float>(ko) : 0.0f;
    for (uint32_t i : in.indices()) {
      float* row = w.data() + static_cast<size_t>(i) * N;
      for (uint32_t j : out.indices()) {
        float& v = row[j];
        switch (rule) {
          case 0:  // hebb
            v += xi * (1.0f - v);
            break;
          case 1:  // covariance (mean subtracted on both sides)
            v += xi * (1.0f - mx) * (1.0f - my);
            break;
          case 2:  // bounded delta gate
            v = v * (1.0f - xi) + xi;
            break;
          default:  // capped hebb
            v += xi * (1.0f - v);
            if (v > cap) v = cap;
            break;
        }
      }
    }
  }

  SDR recall(const SDR& ctx, size_t k) const {
    std::vector<std::pair<float, uint32_t>> s(N, {0.0f, 0u});
    for (size_t j = 0; j < N; ++j) s[j].second = static_cast<uint32_t>(j);
    for (uint32_t i : ctx.indices()) {
      const float* row = w.data() + static_cast<size_t>(i) * N;
      for (size_t j = 0; j < N; ++j) s[j].first += row[j];
    }
    std::partial_sort(s.begin(), s.begin() + static_cast<long>(std::min(k, N)), s.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    SDR out(N, 0);
    for (size_t i = 0; i < k && i < s.size(); ++i) out.set(s[i].second);
    return out;
  }
};

struct Cfg {
  const char* name;
  int rule;
  float xi;
  float cap;
};

}  // namespace

int main() {
  const size_t N = 1024, K = 24;
  const Cfg cfgs[] = {
      {"hebb_0.05", 0, 0.05f, 1.0f},   {"hebb_0.005", 0, 0.005f, 1.0f},
      {"cov_0.05", 1, 0.05f, 1.0f},    {"cov_0.005", 1, 0.005f, 1.0f},
      {"cov_0.001", 1, 0.001f, 1.0f},  {"delta_0.05", 2, 0.05f, 1.0f},
      {"delta_0.005", 2, 0.005f, 1.0f},{"capped_0.5", 3, 0.005f, 0.5f},
  };

  // ---- skewed (Zipfian) world: the DC mode that collapses the real corpus ----
  {
    const size_t V = 4000;
    std::mt19937_64 vr(7);
    // bit i is hot with probability ~ 1/(i+1): a few bits carry most of the traffic
    std::vector<double> bp(N);
    double bz = 0;
    for (size_t i = 0; i < N; ++i) { bp[i] = 1.0 / static_cast<double>(i + 1); bz += bp[i]; }
    std::discrete_distribution<size_t> bit_pick(bp.begin(), bp.end());

    std::vector<SDR> vocab(V);
    for (size_t v = 0; v < V; ++v) {
      SDR c(N, 0);
      while (c.count() < K) c.set(static_cast<uint32_t>(bit_pick(vr)));
      vocab[v] = c;
    }
    std::vector<double> p(V);
    for (size_t v = 0; v < V; ++v) p[v] = 1.0 / std::pow(static_cast<double>(v + 1), 0.9);
    std::discrete_distribution<size_t> word_pick(p.begin(), p.end());

    printf("rule,zipf_load,recall_frequent,recall_rare,mean_w,frac_gt_0.9\n");
    for (const auto& c : cfgs) {
      for (int items : {2000, 8000, 32000}) {
        std::mt19937_64 g(100 + static_cast<uint64_t>(items));
        Mem m(N, c.rule, c.xi, c.cap);
        std::vector<std::pair<int, int>> pairs;
        pairs.reserve(static_cast<size_t>(items));
        for (int i = 0; i < items; ++i) {
          int a = static_cast<int>(word_pick(g));
          int b = static_cast<int>(word_pick(g));
          pairs.emplace_back(a, b);
          m.learn(vocab[static_cast<size_t>(a)], vocab[static_cast<size_t>(b)]);
        }
        double fq = 0, rare = 0;
        int nf = 0, nr = 0;
        for (const auto& [a, b] : pairs) {
          double r = double(m.recall(vocab[static_cast<size_t>(a)], K).overlap(vocab[static_cast<size_t>(b)])) / K;
          if (a < 200) { fq += r; ++nf; }
          if (a >= static_cast<int>(V) - 200) { rare += r; ++nr; }
        }
        double sum = 0, gt9 = 0;
        for (float v : m.w) { sum += v; if (v > 0.9f) ++gt9; }
        const double n = static_cast<double>(m.w.size());
        printf("%s,%d,%.4f,%.4f,%.4f,%.4f\n", c.name, items, nf ? fq / nf : 0.0,
               nr ? rare / nr : 0.0, sum / n, gt9 / n);
      }
    }
  }

  printf("rule,load,recall,mean_w,frac_gt_0.9\n");
  for (const auto& c : cfgs) {
    for (int items : {500, 2000, 8000, 32000}) {
      Rng rng(100 + static_cast<uint64_t>(items));
      std::vector<SDR> keys, vals;
      keys.reserve(static_cast<size_t>(items));
      vals.reserve(static_cast<size_t>(items));
      for (int i = 0; i < items; ++i) {
        keys.push_back(SDR::random(N, K, rng.next()));
        vals.push_back(SDR::random(N, K, rng.next()));
      }
      Mem m(N, c.rule, c.xi, c.cap);
      for (int i = 0; i < items; ++i) m.learn(keys[static_cast<size_t>(i)], vals[static_cast<size_t>(i)]);

      size_t probe = std::min<size_t>(static_cast<size_t>(items), 300);
      double acc = 0;
      for (size_t i = 0; i < probe; ++i)
        acc += double(m.recall(keys[i], K).overlap(vals[i])) / double(K);
      acc /= static_cast<double>(probe);

      double sum = 0, gt9 = 0;
      for (float v : m.w) {
        sum += v;
        if (v > 0.9f) ++gt9;
      }
      const double n = static_cast<double>(m.w.size());
      printf("%s,%d,%.4f,%.4f,%.4f\n", c.name, items, acc, sum / n, gt9 / n);
    }
  }
  return 0;
}