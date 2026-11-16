// Sparse distributed memory (Kanerva hard locations) as the cheap replacement for the
// dense associative matrix. Every line is one measurement, so the report can be
// regenerated exactly: ./build_sdm/bin/sdm
//
// Questions, in the order of docs/24_evidence_protocol.md:
//   1 capacity       recall vs number of stored patterns, plus a dense-matrix baseline
//   2 retention      train list A, measure A, train list B, measure A again
//   3 generalisation recall from a noisy cue (10/25/50 % of the active bits replaced)
//   4 RAM            SDM bytes vs bm::AssociativeMemory (N*N floats) at the same width
//   5 speed          writes/s and reads/s
//   6 control        uncorrelated cue, cross-talk, shuffled pairing, literal half rule
//   6b correlated    correlated codes: SDM vs dense matrix
//   7 attention      content-addressable (key, value) store: rank of the correct value
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/sdm.hpp"
#include "bm/sdr.hpp"

using namespace bm;

namespace {

constexpr size_t kN = 1024;  // SDR width
constexpr size_t kK = 40;    // active bits (3.9 %)
constexpr size_t kA = 4096;  // hard locations
constexpr size_t kC = 1024;  // address / data width
constexpr size_t kT = 28;    // address overlap threshold
constexpr uint32_t kSeed = 0x5D1E5A17u;

SdmParams Params() { return SdmParams{kA, kC, kT, kSeed}; }

std::vector<SDR> MakePatterns(size_t n, size_t k, uint64_t seed0) {
  std::vector<SDR> out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) out.push_back(SDR::random(kN, k, seed0 + i));
  return out;
}

double Recall(const SDR& got, const SDR& want) {
  return want.count() ? static_cast<double>(got.overlap(want)) / static_cast<double>(want.count())
                      : 0.0;
}

double Precision(const SDR& got, const SDR& want) {
  return got.count() ? static_cast<double>(got.overlap(want)) / static_cast<double>(got.count())
                     : 0.0;
}

double MeanRecall(const Sdm& sdm, const std::vector<SDR>& pats) {
  double sum = 0.0;
  for (const SDR& p : pats) sum += Recall(sdm.read(p), p);
  return pats.empty() ? 0.0 : sum / static_cast<double>(pats.size());
}

// Replace `frac` of the active bits with random positions, keeping k fixed. Flipping a
// random 10 % of all N bits would turn a 4 % code into a 50 % code, which is not the
// noise an SDR is supposed to survive.
SDR Noisy(const SDR& s, double frac, Rng& rng) {
  std::vector<uint32_t> idx = s.indices();
  const size_t drop = static_cast<size_t>(frac * static_cast<double>(idx.size()));
  for (size_t i = 0; i < drop && i < idx.size(); ++i) {
    const size_t j = i + rng.below(static_cast<uint32_t>(idx.size() - i));
    std::swap(idx[i], idx[j]);
  }
  SDR out(s.size(), 0);
  for (size_t i = drop; i < idx.size(); ++i) out.set(idx[i]);
  size_t added = 0;
  while (added < drop) {
    const uint32_t p = static_cast<uint32_t>(rng.below(static_cast<uint32_t>(s.size())));
    if (!out.test(p)) {
      out.set(p);
      ++added;
    }
  }
  return out;
}

// ---------------------------------------------------------------- 1 capacity
void Capacity() {
  const size_t counts[] = {10, 50, 100, 200, 400, 800, 1200, 1400, 1500, 1600, 2000, 2400};
  size_t best = 0;
  for (size_t m : counts) {
    Sdm sdm(Params());
    std::vector<SDR> pats = MakePatterns(m, kK, 1000);
    for (const SDR& p : pats) sdm.write(p, p);
    double rec = 0.0, prec = 0.0, bits = 0.0, rec_half = 0.0;
    for (const SDR& p : pats) {
      const SDR r = sdm.read(p);
      rec += Recall(r, p);
      prec += Precision(r, p);
      bits += static_cast<double>(r.count());
      rec_half += Recall(sdm.read(p, SdmRead::HalfSelected), p);
    }
    const double n = static_cast<double>(m);
    std::printf(
        "sdm,capacity,patterns=%zu,recall=%.3f,precision=%.3f,read_bits=%.1f,half_rule_recall=%.3f\n",
        m, rec / n, prec / n, bits / n, rec_half / n);
    if (rec / n >= 0.9) best = m;
  }
  std::printf("sdm,capacity,max_patterns_recall_0.9=%zu\n", best);
  std::printf("sdm,capacity,chance_overlap=%.4f\n",
              static_cast<double>(kK) / static_cast<double>(kN));
}

// Dense baseline at the same width: N*N floats, recall measured the same way.
void CapacityDense() {
  const size_t counts[] = {100, 200, 400, 800, 1200, 1600, 2000, 2400, 3200};
  for (size_t m : counts) {
    AssociativeMemory am(kN, kN, kK);
    std::vector<SDR> pats = MakePatterns(m, kK, 1000);
    std::vector<std::pair<SDR, SDR>> pairs;
    pairs.reserve(m);
    for (const SDR& p : pats) pairs.emplace_back(p, p);
    am.learn_batch(pairs);
    // recall on a 200-pattern sample: the full sweep is O(M * N^2) and the mean is
    // already stable at 200 samples
    const size_t stride = std::max<size_t>(1, m / 200);
    double rec = 0.0, prec = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < m; i += stride) {
      const SDR r = am.recall(pats[i]);
      rec += Recall(r, pats[i]);
      prec += Precision(r, pats[i]);
      ++n;
    }
    std::printf("sdm,dense_capacity,patterns=%zu,recall=%.3f,precision=%.3f,samples=%zu\n", m,
                rec / static_cast<double>(n), prec / static_cast<double>(n), n);
  }
}

// ---------------------------------------------------------------- 2 retention
void Retention() {
  Sdm sdm(Params());
  std::vector<SDR> a = MakePatterns(100, kK, 2000);
  std::vector<SDR> b = MakePatterns(100, kK, 3000);
  for (const SDR& p : a) sdm.write(p, p);
  const double before = MeanRecall(sdm, a);
  for (const SDR& p : b) sdm.write(p, p);
  const double after = MeanRecall(sdm, a);
  const double b_recall = MeanRecall(sdm, b);
  std::printf("sdm,retention,before=%.3f,after_B=%.3f,listB_recall=%.3f,chance=%.3f\n", before,
              after, b_recall, static_cast<double>(kK) / static_cast<double>(kN));
}

// ---------------------------------------------------------------- 3 noise
void Noise() {
  const double levels[] = {0.0, 0.10, 0.25, 0.50};
  Sdm sdm(Params());
  std::vector<SDR> pats = MakePatterns(100, kK, 4000);
  for (const SDR& p : pats) sdm.write(p, p);
  for (double f : levels) {
    Rng rng(5000 + static_cast<uint64_t>(f * 1000.0));
    double rec = 0.0, prec = 0.0, bits = 0.0, rec_half = 0.0;
    for (const SDR& p : pats) {
      const SDR cue = Noisy(p, f, rng);
      const SDR r = sdm.read(cue);
      rec += Recall(r, p);
      prec += Precision(r, p);
      bits += static_cast<double>(r.count());
      rec_half += Recall(sdm.read(cue, SdmRead::HalfSelected), p);
    }
    const double n = static_cast<double>(pats.size());
    std::printf(
        "sdm,noise,flip_frac=%.2f,recall=%.3f,precision=%.3f,read_bits=%.1f,half_rule_recall=%.3f\n",
        f, rec / n, prec / n, bits / n, rec_half / n);
    // dense baseline under the same cues
    AssociativeMemory am(kN, kN, kK);
    std::vector<std::pair<SDR, SDR>> batch;
    batch.reserve(pats.size());
    for (const SDR& p : pats) batch.emplace_back(p, p);
    am.learn_batch(batch);
    Rng rng2(5000 + static_cast<uint64_t>(f * 1000.0));
    double rec2 = 0.0;
    for (const SDR& p : pats) rec2 += Recall(am.recall(Noisy(p, f, rng2)), p);
    std::printf("sdm,noise,model=dense,flip_frac=%.2f,recall=%.3f\n", f, rec2 / n);
  }
}

// ---------------------------------------------------------------- 4 RAM
void Ram() {
  Sdm sdm(Params());
  AssociativeMemory am(kN, kN, kK);
  const size_t sdm_bytes = sdm.ram_bytes();
  const size_t dense_bytes = am.in_size() * am.out_size() * sizeof(float);
  std::printf("sdm,ram,sdm_bytes=%zu,address_bytes=%zu,register_bytes=%zu,count_bytes=%zu\n",
              sdm_bytes, sdm.address_bytes(), sdm.register_bytes(), sdm.count_bytes());
  std::printf("sdm,ram,dense_bytes=%zu,dense_formula=N*N*4,ratio=%.2f\n", dense_bytes,
              static_cast<double>(dense_bytes) / static_cast<double>(sdm_bytes));
  std::printf("sdm,ram,bytes_per_pattern_sdm=%.0f,bytes_per_pattern_dense=%.0f\n",
              static_cast<double>(sdm_bytes) / 100.0, static_cast<double>(dense_bytes) / 100.0);
}

// ---------------------------------------------------------------- 5 speed
void Speed() {
  Sdm sdm(Params());
  std::vector<SDR> pats = MakePatterns(100, kK, 6000);
  for (const SDR& p : pats) sdm.write(p, p);
  const int iters = 500;
  volatile size_t sink = 0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    sdm.write(pats[static_cast<size_t>(i) % pats.size()],
              pats[static_cast<size_t>(i + 1) % pats.size()]);
  }
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    sink = sink + sdm.read(pats[static_cast<size_t>(i) % pats.size()]).count();
  }
  const auto t2 = std::chrono::steady_clock::now();
  (void)sink;

  const double ws = std::chrono::duration<double>(t1 - t0).count();
  const double rs = std::chrono::duration<double>(t2 - t1).count();
  std::printf("sdm,speed,writes_per_s=%.0f,reads_per_s=%.0f,write_us=%.1f,read_us=%.1f\n",
              static_cast<double>(iters) / ws, static_cast<double>(iters) / rs, ws * 1e6 / iters,
              rs * 1e6 / iters);

  // dense baseline, same width: learn_batch amortises the decay, recall is one pass
  AssociativeMemory am(kN, kN, kK);
  std::vector<std::pair<SDR, SDR>> batch;
  batch.reserve(pats.size());
  for (const SDR& p : pats) batch.emplace_back(p, p);
  const auto d0 = std::chrono::steady_clock::now();
  am.learn_batch(batch);
  const auto d1 = std::chrono::steady_clock::now();
  size_t acc2 = 0;
  for (int i = 0; i < iters; ++i) acc2 += am.recall(pats[static_cast<size_t>(i) % pats.size()]).count();
  const auto d2 = std::chrono::steady_clock::now();
  (void)acc2;
  const double dlearn = std::chrono::duration<double>(d1 - d0).count();
  const double drecall = std::chrono::duration<double>(d2 - d1).count();
  std::printf("sdm,speed,model=dense,learn_us_per_pattern=%.1f,recall_us=%.1f\n",
              dlearn * 1e6 / static_cast<double>(pats.size()), drecall * 1e6 / iters);
}

// ---------------------------------------------------------------- 6 controls
void Controls() {
  Sdm sdm(Params());
  std::vector<SDR> keys = MakePatterns(100, kK, 7000);
  std::vector<SDR> vals = MakePatterns(100, kK, 8000);
  for (size_t i = 0; i < keys.size(); ++i) sdm.write(keys[i], vals[i]);

  // (a) a cue that was never stored must return chance-level overlap with any value
  double uncorr = 0.0;
  for (size_t i = 0; i < keys.size(); ++i) {
    const SDR cue = SDR::random(kN, kK, 9500 + i);
    uncorr += Recall(sdm.read(cue), vals[i]);
  }
  std::printf("sdm,control,uncorrelated_cue_recall=%.4f\n", uncorr / 100.0);

  // (b) cross-talk: reading with key i must not return value j (j != i)
  double cross = 0.0;
  size_t n = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    const SDR r = sdm.read(keys[i]);
    for (size_t j = 0; j < vals.size(); ++j) {
      if (i == j) continue;
      cross += Recall(r, vals[j]);
      ++n;
    }
  }
  std::printf("sdm,control,cross_talk_recall=%.4f,pairs=%zu\n", cross / static_cast<double>(n), n);

  // (c) shuffled pairing: write key i with value perm(i); the original partner must not
  // come back. The memory stores the pairing it was shown; it does not reconstruct the
  // data distribution.
  Sdm sh(Params());
  std::vector<uint32_t> perm(vals.size());
  std::iota(perm.begin(), perm.end(), 0u);
  Rng prng(9100);
  for (size_t i = perm.size(); i > 1; --i) {
    const size_t j = prng.below(static_cast<uint32_t>(i));
    std::swap(perm[i - 1], perm[j]);
  }
  for (size_t i = 0; i < keys.size(); ++i) sh.write(keys[i], vals[perm[i]]);
  double hit = 0.0;
  for (size_t i = 0; i < keys.size(); ++i) hit += Recall(sh.read(keys[i]), vals[i]);
  std::printf("sdm,control,shuffled_pair_recall=%.4f\n", hit / 100.0);

  // (d) the literal Kanerva read rule on clean cues
  double bits = 0.0;
  for (size_t i = 0; i < keys.size(); ++i) {
    bits += static_cast<double>(sdm.read(keys[i], SdmRead::HalfSelected).count());
  }
  std::printf("sdm,control,half_rule_bits=%.2f,selected_per_cue=%zu\n", bits / 100.0,
              sdm.selected(keys[0]));
}

// ------------------------------------------------- 6b correlated codes
// Real encoder output is correlated: similar inputs give similar SDRs. Random orthogonal
// codes are the dense correlation matrix's best case; this is its worst case.
void Correlated() {
  const size_t protos = 20, variants = 5;
  std::vector<SDR> pats, proto;
  std::vector<size_t> pid;
  for (size_t p = 0; p < protos; ++p) {
    proto.push_back(SDR::random(kN, kK, 20000 + p));
    for (size_t v = 0; v < variants; ++v) {
      Rng rng(30000 + p * 10 + v);
      pats.push_back(Noisy(proto.back(), 0.30, rng));
      pid.push_back(p);
    }
  }
  Sdm sdm(Params());
  for (const SDR& p : pats) sdm.write(p, p);
  double rec = 0.0, prec = 0.0, bits = 0.0, proto_ov = 0.0;
  for (size_t i = 0; i < pats.size(); ++i) {
    const SDR r = sdm.read(pats[i]);
    rec += Recall(r, pats[i]);
    prec += Precision(r, pats[i]);
    bits += static_cast<double>(r.count());
    proto_ov += Recall(r, proto[pid[i]]);
  }
  const double n = static_cast<double>(pats.size());
  std::printf(
      "sdm,correlated,model=sdm,patterns=%zu,variants=%zu,flip=0.30,recall=%.3f,precision=%.3f,"
      "read_bits=%.1f,prototype_overlap=%.3f\n",
      pats.size(), variants, rec / n, prec / n, bits / n, proto_ov / n);

  AssociativeMemory am(kN, kN, kK);
  std::vector<std::pair<SDR, SDR>> batch;
  batch.reserve(pats.size());
  for (const SDR& p : pats) batch.emplace_back(p, p);
  am.learn_batch(batch);
  double rec2 = 0.0, prec2 = 0.0;
  for (const SDR& p : pats) {
    const SDR r = am.recall(p);
    rec2 += Recall(r, p);
    prec2 += Precision(r, p);
  }
  std::printf("sdm,correlated,model=dense,patterns=%zu,variants=%zu,flip=0.30,recall=%.3f,precision=%.3f\n",
              pats.size(), variants, rec2 / n, prec2 / n);
}

// ---------------------------------------------------------------- 7 attention
int RankOf(const SDR& got, const std::vector<SDR>& values, size_t correct) {
  if (got.count() == 0) return static_cast<int>(values.size());
  const size_t own = got.overlap(values[correct]);
  int rank = 1;
  for (size_t j = 0; j < values.size(); ++j) {
    if (j == correct) continue;
    if (got.overlap(values[j]) > own) ++rank;
  }
  return rank;
}

void AttentionPair(size_t pairs, double noise) {
  std::vector<SDR> keys = MakePatterns(pairs, kK, 10000);
  std::vector<SDR> vals = MakePatterns(pairs, kK, 11000);

  Sdm sdm(Params());
  for (size_t i = 0; i < pairs; ++i) sdm.write(keys[i], vals[i]);
  Rng rng(12000);
  double rank_sum = 0.0, val_rec = 0.0;
  size_t top1 = 0, top5 = 0;
  std::vector<int> ranks;
  for (size_t i = 0; i < pairs; ++i) {
    const SDR r = sdm.read(Noisy(keys[i], noise, rng));
    const int rank = RankOf(r, vals, i);
    ranks.push_back(rank);
    rank_sum += static_cast<double>(rank);
    val_rec += Recall(r, vals[i]);
    if (rank == 1) ++top1;
    if (rank <= 5) ++top5;
  }
  std::sort(ranks.begin(), ranks.end());
  std::printf(
      "sdm,attention,model=sdm,pairs=%zu,noise=%.2f,top1=%.2f,top5=%.2f,mean_rank=%.1f,"
      "median_rank=%d,value_recall=%.3f,chance_top1=%.3f\n",
      pairs, noise, static_cast<double>(top1) / static_cast<double>(pairs),
      static_cast<double>(top5) / static_cast<double>(pairs),
      rank_sum / static_cast<double>(pairs), ranks[ranks.size() / 2],
      val_rec / static_cast<double>(pairs), 1.0 / static_cast<double>(pairs));

  AssociativeMemory am(kN, kN, kK);
  std::vector<std::pair<SDR, SDR>> batch;
  batch.reserve(pairs);
  for (size_t i = 0; i < pairs; ++i) batch.emplace_back(keys[i], vals[i]);
  am.learn_batch(batch);
  Rng rng2(12000);
  double rank_sum2 = 0.0, val_rec2 = 0.0;
  size_t top1_2 = 0, top5_2 = 0;
  std::vector<int> ranks2;
  for (size_t i = 0; i < pairs; ++i) {
    const SDR r = am.recall(Noisy(keys[i], noise, rng2));
    const int rank = RankOf(r, vals, i);
    ranks2.push_back(rank);
    rank_sum2 += static_cast<double>(rank);
    val_rec2 += Recall(r, vals[i]);
    if (rank == 1) ++top1_2;
    if (rank <= 5) ++top5_2;
  }
  std::sort(ranks2.begin(), ranks2.end());
  std::printf(
      "sdm,attention,model=dense,pairs=%zu,noise=%.2f,top1=%.2f,top5=%.2f,mean_rank=%.1f,"
      "median_rank=%d,value_recall=%.3f,chance_top1=%.3f\n",
      pairs, noise, static_cast<double>(top1_2) / static_cast<double>(pairs),
      static_cast<double>(top5_2) / static_cast<double>(pairs),
      rank_sum2 / static_cast<double>(pairs), ranks2[ranks2.size() / 2],
      val_rec2 / static_cast<double>(pairs), 1.0 / static_cast<double>(pairs));

  // Control: a key that was never stored must rank the correct value at chance.
  size_t top1_c = 0;
  for (size_t i = 0; i < pairs; ++i) {
    const SDR r = sdm.read(SDR::random(kN, kK, 13500 + i));
    if (RankOf(r, vals, i) == 1) ++top1_c;
  }
  std::printf("sdm,attention,model=sdm,control=random_key,pairs=%zu,top1=%.2f\n", pairs,
              static_cast<double>(top1_c) / static_cast<double>(pairs));
}

}  // namespace

int main() {
  std::printf("sdm,config,addresses=%zu,width=%zu,threshold=%zu,sdr_n=%zu,sdr_k=%zu\n", kA, kC, kT,
              kN, kK);
  Capacity();
  CapacityDense();
  Retention();
  Noise();
  Ram();
  Speed();
  Controls();
  Correlated();
  AttentionPair(100, 0.10);
  AttentionPair(100, 0.25);
  AttentionPair(400, 0.25);
  return 0;
}