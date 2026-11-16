// Standard evidence bench: does each module actually learn, retain, generalise,
// and stay small? Every section has a control or a baseline.
// Output is one line per measurement so the report can be regenerated exactly.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/encoder.hpp"
#include "bm/spatial_pooler.hpp"
#include "bm/sdr.hpp"
#include "bm/temporal_memory.hpp"

using namespace bm;

namespace {

size_t PeakRssMb() {
  std::ifstream st("/proc/self/status");
  std::string line;
  while (std::getline(st, line)) {
    if (line.rfind("VmHWM:", 0) == 0) return static_cast<size_t>(std::stol(line.substr(6)) / 1024);
  }
  return 0;
}

double MeanOverlap(const SDR& a, const SDR& b) {
  return a.count() ? double(a.overlap(b)) / a.count() : 0.0;
}

SDR Flip(const SDR& s, int bits, Rng& rng) {
  SDR o = s;
  auto idx = s.indices();
  for (int i = 0; i < bits && !idx.empty(); ++i) {
    uint32_t p = idx[static_cast<size_t>(rng.below(static_cast<uint32_t>(idx.size())))];
    if (o.test(p)) {
      o.reset(p);
    } else {
      o.set(p);
    }
  }
  return o;
}

// ---------------------------------------------------------------- SP
void BenchSp() {
  const size_t N = 1024, K = 40;
  Encoder enc(N, K, 3);
  Rng rng(17);
  const int classes = 20, per_class = 20;
  std::vector<SDR> proto;
  for (int c = 0; c < classes; ++c) proto.push_back(SDR::random(N, K, 100 + static_cast<uint64_t>(c)));

  std::vector<std::pair<SDR, int>> train, test;
  for (int c = 0; c < classes; ++c) {
    for (int i = 0; i < per_class; ++i) {
      SDR s = Flip(proto[static_cast<size_t>(c)], 6, rng);
      (i % 2 ? test : train).emplace_back(s, c);
    }
  }

  SpatialPoolerParams pp;
  pp.input_size = N;
  pp.column_count = 512;
  pp.synapses_per_column = 64;
  pp.active_columns = 16;
  pp.seed = 7;
  SpatialPooler sp(pp);
  for (int ep = 0; ep < 3; ++ep) {
    for (const auto& [x, y] : train) {
      (void)y;
      sp.activate(x);
    }
  }

  // stability and noise robustness
  double same = 0, noise1 = 0, noise8 = 0, noise16 = 0;
  for (const auto& [x, y] : test) {
    (void)y;
    SDR a = sp.activate(x);
    same += MeanOverlap(a, sp.activate(x));
    noise1 += MeanOverlap(a, sp.activate(Flip(x, 1, rng)));
    noise8 += MeanOverlap(a, sp.activate(Flip(x, 8, rng)));
    noise16 += MeanOverlap(a, sp.activate(Flip(x, 16, rng)));
  }
  size_t n = test.size();
  printf("sp,stability=%.3f,noise1=%.3f,noise8=%.3f,noise16=%.3f\n", same / n, noise1 / n,
         noise8 / n, noise16 / n);

  // class information after SP vs input directly
  auto accuracy = [&](bool use_sp) {
    AssociativeMemory read(use_sp ? 512 : N, classes, 1, 0.08f);
    for (int ep = 0; ep < 4; ++ep) {
      for (const auto& [x, y] : train) {
        SDR code = use_sp ? sp.activate(x) : x;
        SDR lab(classes, 0);
        lab.set(static_cast<uint32_t>(y));
        read.learn(code, lab);
      }
    }
    int ok = 0;
    for (const auto& [x, y] : test) {
      SDR code = use_sp ? sp.activate(x) : x;
      SDR lab = read.recall(code);
      auto idx = lab.indices();
      if (!idx.empty() && static_cast<int>(idx[0]) == y) ++ok;
    }
    return double(ok) / test.size();
  };
  printf("sp,classinfo_input=%.3f,classinfo_sp=%.3f\n", accuracy(false), accuracy(true));

  // separation: mean pairwise overlap of SP codes for different classes
  std::vector<SDR> codes;
  for (int c = 0; c < classes; ++c) codes.push_back(sp.activate(proto[static_cast<size_t>(c)]));
  double same_c = 0;
  int ns = 0;
  for (size_t i = 0; i < codes.size(); ++i) {
    for (size_t j = i + 1; j < codes.size(); ++j) {
      same_c += MeanOverlap(codes[i], codes[j]);
      ++ns;
    }
  }
  printf("sp,pairwise_code_overlap=%.3f,dead_columns=%zu,syn_per_col=%.1f\n", ns ? same_c / ns : 0,
         sp.dead_columns(), double(sp.mean_synapses()));

  size_t bytes = sp.column_count() * pp.synapses_per_column * sizeof(Syn);
  printf("sp,ram_matrix_kb=%zu\n", bytes / 1024);
}

// ---------------------------------------------------------------- TM
void BenchTm() {
  const size_t N = 512, K = 20;
  const int symbols = 64;
  Encoder enc(N, K, 9);
  std::vector<SDR> sym;
  char name[16];
  for (int s = 0; s < symbols; ++s) {
    std::snprintf(name, sizeof name, "s%d", s);
    sym.push_back(enc.symbol(name));
  }

  // order-1 source: next symbol depends only on the previous symbol (64 contexts)
  std::vector<std::vector<int>> trans(static_cast<size_t>(symbols));
  Rng rng(23);
  for (size_t i = 0; i < trans.size(); ++i) {
    for (int j = 0; j < 3; ++j) trans[i].push_back(static_cast<int>(rng.below(symbols)));
  }
  auto next_of = [&](int, int b) {
    const auto& v = trans[static_cast<size_t>(b)];
    return v[static_cast<size_t>(rng.below(static_cast<uint32_t>(v.size())))];
  };

  TemporalMemoryParams tp;
  tp.column_count = N;
  tp.cells_per_column = 8;
  tp.max_segments_per_cell = 8;
  tp.activation_threshold = 6;
  tp.new_segment_min = 6;
  tp.seed = 11;
  TemporalMemory tm(tp);

  int a = 0, b = 1;
  const int train_steps = 30000;
  double burst_clean = 0, burst_bad = 0;
  for (int t = 0; t < train_steps; ++t) {
    int c = next_of(a, b);
    auto st = tm.step(sym[static_cast<size_t>(c)]);
    burst_clean += double(st.bursting_columns) / double(K);
    a = b;
    b = c;
  }
  // corrupted transitions
  int bad = 0;
  for (int t = 0; t < 2000; ++t) {
    int c = static_cast<int>(rng.below(symbols));
    auto st = tm.step(sym[static_cast<size_t>(c)]);
    burst_bad += double(st.bursting_columns) / double(K);
    ++bad;
    a = b;
    b = c;
  }

  // prediction accuracy on a fresh stream
  int hits = 0, n = 0;
  double base_prev = 0, base_freq = 0;
  std::vector<int> counts(symbols, 0);
  a = 0;
  b = 1;
  for (int t = 0; t < 8000; ++t) {
    int c = next_of(a, b);
    auto st = tm.step(sym[static_cast<size_t>(c)]);
    SDR pred(N, 0);
    for (uint32_t cell : st.predictive.indices()) {
      pred.set(static_cast<uint32_t>(cell / tp.cells_per_column));
    }
    SDR actual = sym[static_cast<size_t>(c)];
    double ov = MeanOverlap(pred, actual);
    hits += ov > 0.5 ? 1 : 0;
    ++n;
    base_prev += MeanOverlap(sym[static_cast<size_t>(b)], actual) > 0.5 ? 1 : 0;
    ++counts[static_cast<size_t>(b)];
    a = b;
    b = c;
  }
  int best = static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());
  for (int t = 0; t < 2000; ++t) {
    int c = next_of(a, b);
    base_freq += MeanOverlap(sym[static_cast<size_t>(best)], sym[static_cast<size_t>(c)]) > 0.5 ? 1 : 0;
    a = b;
    b = c;
  }

  printf("tm,predict_acc=%.3f,prev_baseline=%.3f,freq_baseline=%.3f\n", double(hits) / n,
         base_prev / 8000.0, base_freq / 2000.0);
  printf("tm,burst_frac_clean=%.4f,burst_frac_corrupt=%.4f,segments=%zu\n",
         burst_clean / train_steps, burst_bad / bad, tm.segment_count());
  size_t seg_bytes = 0;
  for (const auto& cells : tm.segments()) {
    for (const auto& s : cells) seg_bytes += s.cells.size() * sizeof(uint32_t) + s.perms.size();
  }
  printf("tm,ram_segments_kb=%zu\n", seg_bytes / 1024);
}

// ---------------------------------------------------------------- associative memory
void BenchAssoc() {
  const size_t N = 512, K = 24;
  Encoder enc(N, K, 13);
  Rng rng(31);

  auto recall_overlap = [&](int n_pat) {
    AssociativeMemory m(N, N, K, 0.05f);
    std::vector<std::pair<SDR, SDR>> pairs;
    for (int i = 0; i < n_pat; ++i) {
      SDR x = SDR::random(N, K, 1000 + static_cast<uint64_t>(i));
      SDR y = SDR::random(N, K, 5000 + static_cast<uint64_t>(i));
      pairs.emplace_back(x, y);
      m.learn(x, y);
    }
    double ov = 0;
    for (const auto& [x, y] : pairs) ov += MeanOverlap(y, m.recall(x));
    return ov / pairs.size();
  };
  for (int p : {10, 50, 100, 200, 400}) {
    printf("assoc,capacity,patterns=%d,recall=%.3f\n", p, recall_overlap(p));
  }

  // retention: A then B, measure A
  AssociativeMemory m(N, N, K, 0.05f);
  std::vector<std::pair<SDR, SDR>> A, B;
  for (int i = 0; i < 100; ++i) {
    A.emplace_back(SDR::random(N, K, 2000 + static_cast<uint64_t>(i)),
                   SDR::random(N, K, 7000 + static_cast<uint64_t>(i)));
    B.emplace_back(SDR::random(N, K, 3000 + static_cast<uint64_t>(i)),
                   SDR::random(N, K, 8000 + static_cast<uint64_t>(i)));
  }
  for (const auto& [x, y] : A) m.learn(x, y);
  double a0 = 0;
  for (const auto& [x, y] : A) a0 += MeanOverlap(y, m.recall(x));
  a0 /= A.size();
  for (const auto& [x, y] : B) m.learn(x, y);
  double a1 = 0;
  for (const auto& [x, y] : A) a1 += MeanOverlap(y, m.recall(x));
  a1 /= A.size();
  printf("assoc,retention,before=%.3f,after_B=%.3f\n", a0, a1);

  // composition: A->X, B->Y, probe A|B
  AssociativeMemory c(N, N, K, 0.05f);
  SDR a1c = SDR::random(N, K, 11), b1c = SDR::random(N, K, 12);
  SDR x1 = SDR::random(N, K, 13), y1 = SDR::random(N, K, 14);
  c.learn(a1c, x1);
  c.learn(b1c, y1);
  std::vector<uint32_t> both_idx = a1c.indices();
  for (uint32_t i : b1c.indices()) both_idx.push_back(i);
  std::sort(both_idx.begin(), both_idx.end());
  both_idx.erase(std::unique(both_idx.begin(), both_idx.end()), both_idx.end());
  SDR both = SDR::from_indices(N, both_idx);
  SDR composed = c.recall(both, 2 * K);  // a superposition is 2K wide
  double ox = MeanOverlap(x1, composed);
  double oy = MeanOverlap(y1, composed);
  printf("assoc,composition,overlap_X=%.3f,overlap_Y=%.3f\n", ox, oy);

  printf("assoc,ram_matrix_kb=%zu\n", N * N * sizeof(float) / 1024);
}

}  // namespace

int main() {
  printf("== sp ==\n");
  BenchSp();
  printf("== tm ==\n");
  BenchTm();
  printf("== assoc ==\n");
  BenchAssoc();
  printf("peak_rss=%zuMB\n", PeakRssMb());
  return 0;
}