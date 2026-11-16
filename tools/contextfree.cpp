// Context-free retrieval: can a fixed-size associative memory bring old information
// into a new context without a fixed context window?
//
// Every test reports cf,* lines. Controls are mandatory (shuffled pairs, random matrix).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <numeric>
#include <string>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/cleanup.hpp"
#include "bm/sdm.hpp"
#include "bm/sdr.hpp"

using namespace bm;

namespace {

double Recall(const AssociativeMemory& m, const SDR& key, const SDR& val, size_t k = 0) {
  SDR r = k ? m.recall(key, k) : m.recall(key);
  return double(r.overlap(val)) / double(val.count());
}

std::vector<SDR> MakeKeys(size_t n, size_t N, size_t K, Rng& rng) {
  std::vector<SDR> v;
  v.reserve(n);
  for (size_t i = 0; i < n; ++i) v.push_back(SDR::random(N, K, rng.next()));
  return v;
}

std::vector<SDR> MakeVals(size_t n, size_t N, size_t K, Rng& rng) {
  std::vector<SDR> v;
  v.reserve(n);
  for (size_t i = 0; i < n; ++i) v.push_back(SDR::random(N, K, rng.next()));
  return v;
}

// ---------------------------------------------------------------- capacity
void Capacity() {
  const size_t N = 1024, K = 24;
  printf("cf,capacity,note=unique pairs learned in one batch\n");
  for (int items : {100, 500, 1000, 2000, 5000, 10000, 20000}) {
    Rng rng(100 + static_cast<uint64_t>(items));
    auto keys = MakeKeys(static_cast<size_t>(items), N, K, rng);
    auto vals = MakeVals(static_cast<size_t>(items), N, K, rng);
    AssociativeMemory m(N, N, K, 0.05f);
    std::vector<std::pair<SDR, SDR>> batch;
    batch.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) batch.emplace_back(keys[i], vals[i]);
    m.learn_batch(batch);
    double acc = 0;
    size_t probe = std::min<size_t>(keys.size(), 500);
    for (size_t i = 0; i < probe; ++i) acc += Recall(m, keys[i], vals[i]);
    acc /= static_cast<double>(probe);
    printf("cf,capacity,items=%d,recall=%.4f,probe=%zu\n", items, acc, probe);
  }
}

// ---------------------------------------------------------------- age / lag
// Train one item at a time (true online, one decay per item), then ask for items
// of different ages. A fixed window can only answer recent items; a memory should
// degrade gradually.
void AgeCurve() {
  const size_t N = 1024, K = 24;
  const int M = 1500;
  Rng rng(7);
  auto keys = MakeKeys(static_cast<size_t>(M), N, K, rng);
  auto vals = MakeVals(static_cast<size_t>(M), N, K, rng);

  AssociativeMemory m(N, N, K, 0.05f);
  for (int i = 0; i < M; ++i) m.learn(keys[static_cast<size_t>(i)], vals[static_cast<size_t>(i)]);

  const int ages[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1499};
  const int windows[] = {64, 256};

  printf("cf,age,note=recall of an item last seen 'age' items ago (fresh per item)\n");
  for (int age : ages) {
    // average over several items at the same age so the estimate is not a single sample
    double acc = 0;
    int n = 0;
    for (int base = 0; base + age < M && n < 40; base += std::max(1, (M - age) / 40)) {
      int idx = M - 1 - age - base;
      if (idx < 0) continue;
      acc += Recall(m, keys[static_cast<size_t>(idx)], vals[static_cast<size_t>(idx)]);
      ++n;
    }
    if (!n) continue;
    acc /= static_cast<double>(n);
    printf("cf,age,age=%d,recall=%.4f,n=%d", age, acc, n);
    for (int w : windows) printf(",win%d=%.3f", w, age < w ? 1.0 : 0.0);
    printf("\n");
  }
}

// ---------------------------------------------------------------- multi-hop
// Many facts coexist in one memory (distractors + chains). Retrieval must follow a
// chain of old items used as cues, so a deep chain has to survive interference.
void MultiHop() {
  const size_t N = 512, K = 16;
  const int LINKS = 5;
  const int CHAINS = 100;
  const int DISTRACTORS = 500;
  Rng rng(31);

  AssociativeMemory m(N, N, K, 0.05f);
  std::vector<std::pair<SDR, SDR>> batch;
  for (int i = 0; i < DISTRACTORS; ++i)
    batch.emplace_back(SDR::random(N, K, rng.next()), SDR::random(N, K, rng.next()));

  std::vector<std::vector<SDR>> chains;
  for (int c = 0; c < CHAINS; ++c) {
    std::vector<SDR> k = MakeKeys(static_cast<size_t>(LINKS + 1), N, K, rng);
    for (int i = 0; i < LINKS; ++i) batch.emplace_back(k[static_cast<size_t>(i)], k[static_cast<size_t>(i + 1)]);
    chains.push_back(std::move(k));
  }
  m.learn_batch(batch);

  CleanupMemory cm(N, 8192);
  for (const auto& k : chains)
    for (const auto& c : k) cm.add(c);
  for (const auto& [x, y] : batch) {
    cm.add(x);
    cm.add(y);
  }

  printf("cf,hops,note=%d chains of %d links + %d distractors in one memory\n", CHAINS, LINKS,
         DISTRACTORS);
  for (int depth = 1; depth <= LINKS; ++depth) {
    int ok = 0, ok_clean = 0;
    for (const auto& k : chains) {
      for (int mode = 0; mode < 2; ++mode) {
        SDR cur = k[0];
        bool good = true;
        for (int i = 0; i < depth; ++i) {
          SDR r = m.recall(cur);
          if (static_cast<size_t>(r.overlap(k[static_cast<size_t>(i + 1)])) * 2 <
              k[static_cast<size_t>(i + 1)].count()) {
            good = false;
            break;
          }
          cur = mode ? cm.clean(r) : r;
        }
        if (good) {
          if (mode) ++ok_clean;
          else ++ok;
        }
      }
    }
    printf("cf,hops,depth=%d,raw=%.4f,cleaned=%.4f,chains=%d\n", depth, double(ok) / CHAINS,
           double(ok_clean) / CHAINS, CHAINS);
  }

  // per-hop survival: how far does a chain get before it breaks?
  std::vector<int> hist(LINKS + 1, 0);
  for (const auto& k : chains) {
    SDR cur = k[0];
    int reached = 0;
    for (int i = 0; i < LINKS; ++i) {
      SDR r = m.recall(cur);
      if (r.overlap(k[static_cast<size_t>(i + 1)]) * 2 < k[static_cast<size_t>(i + 1)].count()) break;
      cur = r;
      ++reached;
    }
    ++hist[static_cast<size_t>(reached)];
  }
  printf("cf,hops_hist");
  for (int d = 0; d <= LINKS; ++d) printf(",d%d=%d", d, hist[static_cast<size_t>(d)]);
  printf("\n");
  printf("cf,hops_ram,cleanup_items=%zu,cleanup_bytes=%zu\n", cm.size(), cm.bytes());
}

// ---------------------------------------------------------------- new context
// Store (a -> b) and (bundle(b, n) -> v). Query a, get b back, then use b inside a
// NEW context n. Does the old item work as part of a new context?
void NewContext() {
  const size_t N = 1024, K = 24;
  const int CONTEXTS = 4;
  const int PER_CONTEXT = 50;
  const int ITEMS = CONTEXTS * PER_CONTEXT;
  Rng rng(53);
  auto a = MakeKeys(static_cast<size_t>(ITEMS), N, K, rng);
  auto b = MakeKeys(static_cast<size_t>(ITEMS), N, K, rng);
  auto v = MakeVals(static_cast<size_t>(ITEMS), N, K, rng);
  std::vector<SDR> ctx(CONTEXTS);
  for (int c = 0; c < CONTEXTS; ++c) ctx[static_cast<size_t>(c)] = SDR::random(N, 8, rng.next());

  AssociativeMemory reg_a(N, N, K, 0.05f);
  std::vector<std::pair<SDR, SDR>> ba;
  for (int i = 0; i < ITEMS; ++i) ba.emplace_back(a[static_cast<size_t>(i)], b[static_cast<size_t>(i)]);
  reg_a.learn_batch(ba);

  auto cue_of = [&](int i) {
    SDR cue(N, 0);
    for (uint32_t x : b[static_cast<size_t>(i)].indices()) cue.set(x);
    for (uint32_t x : ctx[static_cast<size_t>(i / PER_CONTEXT)].indices()) cue.set(x);
    return cue;
  };

  // one read-out per context: 4 x 1 MB
  std::vector<AssociativeMemory> per_ctx;
  for (int c = 0; c < CONTEXTS; ++c) {
    per_ctx.emplace_back(N, N, K, 0.05f);
    std::vector<std::pair<SDR, SDR>> batch;
    for (int i = c * PER_CONTEXT; i < (c + 1) * PER_CONTEXT; ++i)
      batch.emplace_back(cue_of(i), v[static_cast<size_t>(i)]);
    per_ctx[static_cast<size_t>(c)].learn_batch(batch);
  }

  // control: all contexts in one read-out
  AssociativeMemory shared(N, N, K, 0.05f);
  std::vector<std::pair<SDR, SDR>> all;
  for (int i = 0; i < ITEMS; ++i) all.emplace_back(cue_of(i), v[static_cast<size_t>(i)]);
  shared.learn_batch(all);

  double end2end = 0, direct = 0, wrong = 0, bhit = 0, shared_acc = 0;
  for (int i = 0; i < ITEMS; ++i) {
    int c = i / PER_CONTEXT;
    SDR bhat = reg_a.recall(a[static_cast<size_t>(i)]);
    bhit += double(bhat.overlap(b[static_cast<size_t>(i)])) / b[static_cast<size_t>(i)].count();

    SDR cue(N, 0);
    for (uint32_t x : bhat.indices()) cue.set(x);
    for (uint32_t x : ctx[static_cast<size_t>(c)].indices()) cue.set(x);
    end2end += Recall(per_ctx[static_cast<size_t>(c)], cue, v[static_cast<size_t>(i)]);
    shared_acc += Recall(shared, cue, v[static_cast<size_t>(i)]);

    direct += Recall(per_ctx[static_cast<size_t>(c)], cue_of(i), v[static_cast<size_t>(i)]);

    int j = (i + 1) % ITEMS;
    wrong += Recall(per_ctx[static_cast<size_t>(c)], cue_of(j), v[static_cast<size_t>(i)]);
  }
  printf("cf,new_context,regions=1+4,end_to_end=%.4f,direct_upper=%.4f,wrong_key=%.4f,"
         "retrieved_b=%.4f\n",
         end2end / ITEMS, direct / ITEMS, wrong / ITEMS, bhit / ITEMS);
  printf("cf,new_context,control=one_shared_readout,end_to_end=%.4f,ram_mb=%.1f\n", shared_acc / ITEMS,
         double(5 * N * N * sizeof(float)) / (1024.0 * 1024.0));
}

// ---------------------------------------------------------------- controls
void Controls() {
  const size_t N = 512, K = 16;
  const int ITEMS = 500;
  Rng rng(97);
  auto keys = MakeKeys(static_cast<size_t>(ITEMS), N, K, rng);
  auto vals = MakeVals(static_cast<size_t>(ITEMS), N, K, rng);

  // shuffled values
  AssociativeMemory m(N, N, K, 0.05f);
  std::vector<std::pair<SDR, SDR>> batch;
  for (int i = 0; i < ITEMS; ++i)
    batch.emplace_back(keys[static_cast<size_t>(i)], vals[static_cast<size_t>((i * 7 + 3) % ITEMS)]);
  m.learn_batch(batch);
  double shuf = 0;
  for (int i = 0; i < ITEMS; ++i) shuf += Recall(m, keys[static_cast<size_t>(i)], vals[static_cast<size_t>(i)]);
  printf("cf,control,shuffled_values=%.4f\n", shuf / ITEMS);

  // untrained matrix
  AssociativeMemory empty(N, N, K, 0.05f);
  double rnd = 0;
  for (int i = 0; i < ITEMS; ++i) rnd += Recall(empty, keys[static_cast<size_t>(i)], vals[static_cast<size_t>(i)]);
  printf("cf,control,untrained_matrix=%.4f\n", rnd / ITEMS);
}

// ---------------------------------------------------------------- RAM
void Ram() {
  const size_t N = 512, K = 16;
  AssociativeMemory dense(N, N, K, 0.05f);
  Sdm sdm(SdmParams{4096, 1024, 28, 7});
  printf("cf,ram,dense_bytes=%zu,sdm_bytes=%zu,note=state does not grow with context length\n",
         N * N * sizeof(float), sdm.ram_bytes());
}

}  // namespace

int main() {
  Capacity();
  AgeCurve();
  MultiHop();
  NewContext();
  Controls();
  Ram();
  return 0;
}