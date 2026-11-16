// Cleanup memory: does snapping a retrieved code back to a canonical code fix the
// compounding error of iterated retrieval?

#include <chrono>
#include <cstdio>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/cleanup.hpp"
#include "bm/sdr.hpp"

using namespace bm;

namespace {

std::vector<SDR> Keys(size_t n, size_t N, size_t K, Rng& rng) {
  std::vector<SDR> v;
  v.reserve(n);
  for (size_t i = 0; i < n; ++i) v.push_back(SDR::random(N, K, rng.next()));
  return v;
}

}  // namespace

int main() {
  const size_t N = 512, K = 16;

  // ---------------------------------------------------------------- denoise
  {
    Rng rng(11);
    CleanupMemory cm(N, 4096);
    auto codes = Keys(4096, N, K, rng);
    for (const auto& c : codes) cm.add(c);

    printf("cleanup,denoise,note=flip f bits in a stored code, does clean restore it\n");
    for (int flips : {0, 2, 4, 6, 8, 10, 12, 14, 16}) {
      int exact = 0, trials = 300;
      Rng nr(100 + static_cast<uint64_t>(flips));
      for (int t = 0; t < trials; ++t) {
        SDR noisy = codes[static_cast<size_t>(nr.below(4096))];
        std::vector<uint32_t> off;
        for (uint32_t i = 0; i < N; ++i)
          if (!noisy.test(i)) off.push_back(i);
        for (int f = 0; f < flips && !off.empty(); ++f) {
          size_t j = static_cast<size_t>(nr.below(static_cast<uint32_t>(off.size())));
          noisy.set(off[j]);
          off.erase(off.begin() + static_cast<long>(j));
        }
        SDR got = cm.clean(noisy);
        // compare against the true source code, recovered by nearest
        auto [idx, ov] = cm.nearest(noisy);
        exact += static_cast<size_t>(got.overlap(codes[idx])) == got.count() ? 1 : 0;
      }
      printf("cleanup,denoise,flips=%d,exact=%.3f,trials=%d\n", flips, double(exact) / trials, trials);
    }

    // control: a code that is not in the set must be rejected
    SDR foreign = SDR::random(N, K, rng.next());
    printf("cleanup,control,foreign_code_is_known=%d\n", int(cm.is_known(foreign)));
    printf("cleanup,ram,bytes=%zu,items=%zu\n", cm.bytes(), cm.size());
  }

  // ---------------------------------------------------------------- chains
  {
    const int LINKS = 5, CHAINS = 100, DISTRACTORS = 500;
    Rng rng(31);
    AssociativeMemory mem(N, N, K, 0.05f);
    std::vector<std::pair<SDR, SDR>> batch;
    for (int i = 0; i < DISTRACTORS; ++i)
      batch.emplace_back(SDR::random(N, K, rng.next()), SDR::random(N, K, rng.next()));
    std::vector<std::vector<SDR>> chains;
    for (int c = 0; c < CHAINS; ++c) {
      auto k = Keys(static_cast<size_t>(LINKS + 1), N, K, rng);
      for (int i = 0; i < LINKS; ++i)
        batch.emplace_back(k[static_cast<size_t>(i)], k[static_cast<size_t>(i + 1)]);
      chains.push_back(std::move(k));
    }
    mem.learn_batch(batch);

    CleanupMemory cm(N, 4096);
    for (const auto& ch : chains)
      for (const auto& c : ch) cm.add(c);
    for (const auto& [x, y] : batch) {
      (void)x;
      cm.add(y);
    }

    printf("cleanup,hops,note=%d chains of %d links + %d distractors\n", CHAINS, LINKS,
           DISTRACTORS);
    for (int depth = 1; depth <= LINKS; ++depth) {
      int raw_ok = 0, clean_ok = 0;
      for (const auto& k : chains) {
        for (int mode = 0; mode < 2; ++mode) {
          SDR cur = k[0];
          bool ok = true;
          for (int i = 0; i < depth; ++i) {
            SDR r = mem.recall(cur);
            if (static_cast<size_t>(r.overlap(k[static_cast<size_t>(i + 1)])) * 2 < k[static_cast<size_t>(i + 1)].count()) {
              ok = false;
              break;
            }
            cur = mode ? cm.clean(r) : r;
          }
          if (ok) {
            if (mode) ++clean_ok;
            else ++raw_ok;
          }
        }
      }
      printf("cleanup,hops,depth=%d,raw=%.3f,cleaned=%.3f\n", depth, double(raw_ok) / CHAINS,
             double(clean_ok) / CHAINS);
    }

    // how often does cleanup change the code, and is the change right?
    int changed = 0, changed_right = 0, total = 0;
    for (const auto& k : chains) {
      SDR cur = k[0];
      for (int i = 0; i < LINKS; ++i) {
        SDR r = mem.recall(cur);
        SDR c = cm.clean(r);
        ++total;
        if (static_cast<size_t>(c.overlap(r)) != r.count()) {
          ++changed;
          changed_right += static_cast<size_t>(c.overlap(k[static_cast<size_t>(i + 1)])) ==
                                   c.count()
                               ? 1
                               : 0;
        }
        cur = c;
      }
    }
    printf("cleanup,effect,changed=%d,changed_correct=%d,total=%d\n", changed, changed_right, total);
  }

  // ---------------------------------------------------------------- scale
  {
    const size_t SN = 1024, SK = 24;
    printf("cleanup,scale,note=does denoise hold as the code set grows\n");
    for (int items : {1000, 10000, 100000, 500000}) {
      Rng rng(500 + static_cast<uint64_t>(items));
      CleanupMemory cm(SN, static_cast<size_t>(items));
      std::vector<SDR> codes;
      codes.reserve(static_cast<size_t>(items));
      for (int i = 0; i < items; ++i) codes.push_back(SDR::random(SN, SK, rng.next()));
      for (const auto& c : codes) cm.add(c);
      int trials = 200, exact = 0;
      Rng nr(900 + static_cast<uint64_t>(items));
      for (int t = 0; t < trials; ++t) {
        SDR noisy = codes[static_cast<size_t>(nr.below(static_cast<uint32_t>(items)))];
        for (int f = 0; f < 6; ++f) noisy.reset(static_cast<uint32_t>(nr.below(SN)));
        // flip 6 bits
        for (int f = 0; f < 6; ++f) {
          uint32_t b = static_cast<uint32_t>(nr.below(SN));
          noisy.set(b);
        }
        auto [idx, ov] = cm.nearest(noisy);
        exact += static_cast<size_t>(cm.clean(noisy).overlap(codes[idx])) ==
                         codes[idx].count()
                     ? 1
                     : 0;
      }
      auto t0 = std::chrono::steady_clock::now();
      SDR q = codes[0];
      for (int i = 0; i < 2000; ++i) q = cm.clean(codes[static_cast<size_t>(i % items)]);
      auto t1 = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 2000;
      printf("cleanup,scale,items=%d,exact=%.3f,us_per_clean=%.1f,mb=%.1f\n", items,
             double(exact) / trials, us, double(cm.bytes()) / (1024.0 * 1024.0));
      (void)q;
    }
  }

  // ---------------------------------------------------------------- latency
  {
    Rng rng(77);
    CleanupMemory cm(N, 4096);
    auto codes = Keys(4096, N, K, rng);
    for (const auto& c : codes) cm.add(c);
    SDR q = codes[0];
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 2000;
    for (int i = 0; i < reps; ++i) q = cm.clean(codes[static_cast<size_t>(i % 4096)]);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    printf("cleanup,latency,us_per_clean=%.2f,items=%zu\n", us, cm.size());
    (void)q;
  }

  return 0;
}