// Super-linear read-out for associative memory.
//
// The linear read-out s[j] = sum_i W[i][j] is the lowest member of the dense-associative-
// memory family (Krotov & Hopfield 2016): capacity grows with the power of the read-out
// non-linearity. This tool keeps the same local Hebbian training and only changes the
// read-out to s[j] = sum_i W[i][j]^p, then measures capacity, retention, and the
// shared-context case.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "bm/sdr.hpp"

using namespace bm;

namespace {

struct LinearMem {
  size_t N;
  float lr = 0.05f;
  std::vector<float> w;  // w[i * N + j]

  explicit LinearMem(size_t n) : N(n), w(n * n, 0.0f) {}

  void learn_batch(const std::vector<std::pair<SDR, SDR>>& pairs) {
    for (auto& x : w) x *= 0.99999f;
    for (const auto& [in, out] : pairs) {
      for (uint32_t j : out.indices()) {
        float* col = w.data() + j;
        for (uint32_t i : in.indices()) col[static_cast<size_t>(i) * N] += lr;
      }
    }
  }

  std::vector<float> scores(const SDR& ctx, int p) const {
    std::vector<float> s(N, 0.0f);
    for (uint32_t i : ctx.indices()) {
      const float* row = w.data() + static_cast<size_t>(i) * N;
      for (size_t j = 0; j < N; ++j) {
        float v = row[j];
        if (p == 1) s[j] += v;
        else if (p == 2) s[j] += v * v;
        else {
          float v2 = v * v;
          s[j] += v2 * v;
        }
      }
    }
    return s;
  }

  SDR recall(const SDR& ctx, size_t k, int p) const {
    auto s = scores(ctx, p);
    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(N);
    for (size_t j = 0; j < N; ++j) ranked.emplace_back(s[j], static_cast<uint32_t>(j));
    std::partial_sort(ranked.begin(), ranked.begin() + static_cast<long>(std::min(k, N)),
                      ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    SDR out(N, 0);
    for (size_t i = 0; i < k && i < ranked.size(); ++i) out.set(ranked[i].second);
    return out;
  }
};

double Recall(const LinearMem& m, const SDR& ctx, const SDR& val, size_t k, int p) {
  SDR r = m.recall(ctx, k, p);
  return double(r.overlap(val)) / double(val.count());
}

std::vector<SDR> RandomCodes(size_t n, size_t N, size_t K, Rng& rng) {
  std::vector<SDR> v;
  v.reserve(n);
  for (size_t i = 0; i < n; ++i) v.push_back(SDR::random(N, K, rng.next()));
  return v;
}

}  // namespace

int main() {
  const size_t N = 1024, K = 24;

  printf("power,capacity,note=random pairs, recall at fixed load for read-out power p\n");
  for (int p : {1, 2, 3}) {
    printf("power,p=%d", p);
    for (int items : {500, 1000, 2000, 5000, 10000, 20000}) {
      Rng rng(100 + static_cast<uint64_t>(items));
      auto keys = RandomCodes(static_cast<size_t>(items), N, K, rng);
      auto vals = RandomCodes(static_cast<size_t>(items), N, K, rng);
      LinearMem m(N);
      std::vector<std::pair<SDR, SDR>> batch;
      for (size_t i = 0; i < keys.size(); ++i) batch.emplace_back(keys[i], vals[i]);
      m.learn_batch(batch);
      size_t probe = std::min<size_t>(keys.size(), 300);
      double acc = 0;
      for (size_t i = 0; i < probe; ++i) acc += Recall(m, keys[i], vals[i], K, p);
      acc /= static_cast<double>(probe);
      printf(",%d=%.4f", items, acc);
    }
    printf("\n");
  }

  // shared context: one role tag reused by every pattern (the case where a linear
  // read-out drowned the signal)
  printf("power,shared_context,note=200 pairs share an 8-bit role tag\n");
  for (int p : {1, 2, 3}) {
    Rng rng(53);
    const int ITEMS = 200;
    auto b = RandomCodes(static_cast<size_t>(ITEMS), N, K, rng);
    auto v = RandomCodes(static_cast<size_t>(ITEMS), N, K, rng);
    SDR role = SDR::random(N, 8, rng.next());
    LinearMem m(N);
    std::vector<std::pair<SDR, SDR>> batch;
    for (int i = 0; i < ITEMS; ++i) {
      SDR cue(N, 0);
      for (uint32_t x : b[static_cast<size_t>(i)].indices()) cue.set(x);
      for (uint32_t x : role.indices()) cue.set(x);
      batch.emplace_back(cue, v[static_cast<size_t>(i)]);
    }
    m.learn_batch(batch);
    double acc = 0;
    for (int i = 0; i < ITEMS; ++i) {
      SDR cue(N, 0);
      for (uint32_t x : b[static_cast<size_t>(i)].indices()) cue.set(x);
      for (uint32_t x : role.indices()) cue.set(x);
      acc += Recall(m, cue, v[static_cast<size_t>(i)], K, p);
    }
    printf("power,shared_context,p=%d,recall=%.4f\n", p, acc / ITEMS);
  }

  // retention: A then B, measure A
  printf("power,retention,note=100 pairs A then 100 pairs B, recall A\n");
  for (int p : {1, 2, 3}) {
    Rng rng(91);
    auto ka = RandomCodes(100, N, K, rng);
    auto va = RandomCodes(100, N, K, rng);
    auto kb = RandomCodes(100, N, K, rng);
    auto vb = RandomCodes(100, N, K, rng);
    LinearMem m(N);
    std::vector<std::pair<SDR, SDR>> A, B;
    for (int i = 0; i < 100; ++i) A.emplace_back(ka[static_cast<size_t>(i)], va[static_cast<size_t>(i)]);
    for (int i = 0; i < 100; ++i) B.emplace_back(kb[static_cast<size_t>(i)], vb[static_cast<size_t>(i)]);
    m.learn_batch(A);
    double before = 0;
    for (int i = 0; i < 100; ++i) before += Recall(m, ka[static_cast<size_t>(i)], va[static_cast<size_t>(i)], K, p);
    m.learn_batch(B);
    double after = 0;
    for (int i = 0; i < 100; ++i) after += Recall(m, ka[static_cast<size_t>(i)], va[static_cast<size_t>(i)], K, p);
    printf("power,retention,p=%d,before=%.4f,after=%.4f\n", p, before / 100, after / 100);
  }

  // control: shuffled values must stay at chance for every p
  {
    Rng rng(17);
    auto k = RandomCodes(500, N, K, rng);
    auto v = RandomCodes(500, N, K, rng);
    for (int p : {1, 2, 3}) {
      LinearMem m(N);
      std::vector<std::pair<SDR, SDR>> batch;
      for (size_t i = 0; i < k.size(); ++i)
        batch.emplace_back(k[i], v[(i * 7 + 3) % v.size()]);
      m.learn_batch(batch);
      double acc = 0;
      for (size_t i = 0; i < k.size(); ++i) acc += Recall(m, k[i], v[i], K, p);
      printf("power,control_shuffled,p=%d,recall=%.4f\n", p, acc / k.size());
    }
  }

  return 0;
}