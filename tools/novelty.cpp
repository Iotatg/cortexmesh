// Novelty / anomaly detection bench for the temporal memory.
//
// D26 (docs/13_discoveries.md) found that the raw burst fraction separates clean
// from corrupt input only weakly (0.412 vs 0.521). This tool asks the question
// properly: build an ROC over a mixed stream, one curve per signal x anomaly
// type, with a shuffled-label control, a fixed-FPR operating point, and a
// sequential clean -> corrupt trace.
//
// Setup is copied from BenchTm in tools/bench.cpp (64 symbols, order-1 source,
// 3 successors each, 512 columns x 8 cells, 30k training steps).
//
// Only the public TemporalMemory API is used: step(), predictive(),
// anomaly_score(), reset(), segment_count(), segments(), previous_active().
// Nothing in src/ or include/ is modified.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "bm/encoder.hpp"
#include "bm/sdr.hpp"
#include "bm/temporal_memory.hpp"

using namespace bm;

namespace {

// 0-3 are the required four signals; 4-6 are extra graded / persistence views.
constexpr size_t kNumSignals = 7;
const char* kSignalNames[kNumSignals] = {"burst_fraction", "pred_error", "unexpectedness",
                                         "surprise_ewma", "surprise_slow", "context_mismatch",
                                         "surprise_win20"};
constexpr double kEwmAlpha = 0.2;
constexpr double kSlowAlpha = 0.05;
constexpr size_t kWindow = 20;

size_t PeakRssMb() {
  std::ifstream st("/proc/self/status");
  std::string line;
  while (std::getline(st, line)) {
    if (line.rfind("VmHWM:", 0) == 0) return static_cast<size_t>(std::stol(line.substr(6)) / 1024);
  }
  return 0;
}

struct Sample {
  double s[kNumSignals];
  int label;
};

int PickNext(const std::vector<std::vector<int>>& trans, int b, Rng& rng) {
  const std::vector<int>& v = trans[static_cast<size_t>(b)];
  return v[rng.below(static_cast<uint32_t>(v.size()))];
}

// Rank-based AUC (Mann-Whitney U) with tie correction.
double Auc(const std::vector<double>& score, const std::vector<int>& label) {
  const size_t n = score.size();
  if (n == 0) return 0.5;
  std::vector<size_t> idx(n);
  std::iota(idx.begin(), idx.end(), size_t{0});
  std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) { return score[x] < score[y]; });

  std::vector<double> rank(n, 0.0);
  for (size_t i = 0; i < n;) {
    size_t j = i + 1;
    while (j < n && score[idx[j]] == score[idx[i]]) ++j;
    const double avg = 0.5 * (static_cast<double>(i) + static_cast<double>(j - 1)) + 1.0;
    for (size_t k = i; k < j; ++k) rank[idx[k]] = avg;
    i = j;
  }

  double sum_pos = 0.0;
  size_t npos = 0, nneg = 0;
  for (size_t i = 0; i < n; ++i) {
    if (label[i] > 0) {
      sum_pos += rank[i];
      ++npos;
    } else {
      ++nneg;
    }
  }
  if (npos == 0 || nneg == 0) return 0.5;
  return (sum_pos - static_cast<double>(npos) * (static_cast<double>(npos) + 1.0) / 2.0) /
         (static_cast<double>(npos) * static_cast<double>(nneg));
}

struct OpPoint {
  double threshold = 0.0;
  double tpr = 0.0;
  double fpr = 0.0;
  bool found = false;
};

// Highest TPR whose FPR stays at or below target_fpr.
OpPoint ThresholdAtFpr(const std::vector<double>& clean, const std::vector<double>& pos,
                       double target_fpr) {
  std::vector<double> cs = clean;
  std::vector<double> ps = pos;
  std::sort(cs.begin(), cs.end());
  std::sort(ps.begin(), ps.end());
  OpPoint best;
  for (double t : cs) {
    const size_t fp = static_cast<size_t>(cs.end() - std::lower_bound(cs.begin(), cs.end(), t));
    const size_t tp = static_cast<size_t>(ps.end() - std::lower_bound(ps.begin(), ps.end(), t));
    const double fpr = cs.empty() ? 0.0 : static_cast<double>(fp) / static_cast<double>(cs.size());
    const double tpr = ps.empty() ? 0.0 : static_cast<double>(tp) / static_cast<double>(ps.size());
    if (fpr > target_fpr + 1e-12) continue;
    if (!best.found || tpr > best.tpr + 1e-12 ||
        (std::fabs(tpr - best.tpr) <= 1e-12 && fpr < best.fpr)) {
      best.threshold = t;
      best.tpr = tpr;
      best.fpr = fpr;
      best.found = true;
    }
  }
  return best;
}

// Turns one TMStep plus the actual input into candidate anomaly signals.
// `prev_ctx` is the context the TM used for this step (previous active |
// previous predictive), which lets us grade how well each input column matched.
class Meter {
 public:
  Meter(const TemporalMemory& tm, const TemporalMemoryParams& p, double alpha_fast,
        double alpha_slow)
      : segs_(tm.segments()),
        columns_(p.column_count),
        cells_per_column_(p.cells_per_column),
        perm_threshold_(p.permanence_threshold),
        act_threshold_(p.activation_threshold),
        alpha_fast_(alpha_fast),
        alpha_slow_(alpha_slow),
        win_(kWindow, 0.0),
        pred_col_(p.column_count, 0) {}

  void reset() {
    ewma_fast_ = 0.0;
    ewma_slow_ = 0.0;
    win_.assign(kWindow, 0.0);
    win_n_ = 0;
    win_pos_ = 0;
  }

  void operator()(const TMStep& st, const SDR& input, const SDR& prev_ctx,
                  double out[kNumSignals]) {
    std::fill(pred_col_.begin(), pred_col_.end(), uint8_t{0});
    for (uint32_t cell : st.predictive.indices()) {
      const size_t col = static_cast<size_t>(cell) / cells_per_column_;
      if (col < columns_) pred_col_[col] = 1;
    }

    const size_t n_active = input.count();
    size_t hit = 0;
    for (uint32_t col : input.indices()) {
      if (col < columns_ && pred_col_[col] != 0) ++hit;
    }

    const size_t active_cells = st.active.count();
    out[0] =
        n_active > 0 ? static_cast<double>(st.bursting_columns) / static_cast<double>(n_active) : 0.0;
    out[1] = n_active > 0 ? 1.0 - static_cast<double>(hit) / static_cast<double>(n_active) : 0.0;
    out[2] = active_cells > 0
                 ? 1.0 - static_cast<double>(st.predicted_active) / static_cast<double>(active_cells)
                 : 0.0;

    ewma_fast_ = (1.0 - alpha_fast_) * ewma_fast_ + alpha_fast_ * out[1];
    ewma_slow_ = (1.0 - alpha_slow_) * ewma_slow_ + alpha_slow_ * out[1];
    out[3] = ewma_fast_;
    out[4] = ewma_slow_;

    win_[win_pos_] = out[1];
    win_pos_ = (win_pos_ + 1) % kWindow;
    if (win_n_ < kWindow) ++win_n_;
    double wsum = 0.0;
    for (size_t i = 0; i < win_n_; ++i) wsum += win_[i];
    out[6] = win_n_ > 0 ? wsum / static_cast<double>(win_n_) : 0.0;

    // Graded context match: best active-synapse count over the column's segments,
    // saturating at the activation threshold. 1.0 = every column was strongly
    // predicted, 0.0 = no column matched.
    double sum = 0.0;
    for (uint32_t col : input.indices()) {
      if (col >= columns_) continue;
      size_t best = 0;
      const size_t base = static_cast<size_t>(col) * cells_per_column_;
      for (size_t i = 0; i < cells_per_column_; ++i) {
        for (const Segment& s : segs_[base + i]) {
          size_t a = 0;
          for (size_t k = 0; k < s.cells.size(); ++k) {
            if (s.perms[k] >= perm_threshold_ && prev_ctx.test(s.cells[k])) ++a;
          }
          best = std::max(best, a);
        }
      }
      sum += std::min(1.0, static_cast<double>(best) / static_cast<double>(act_threshold_));
    }
    out[5] = n_active > 0 ? 1.0 - sum / static_cast<double>(n_active) : 0.0;
  }

 private:
  const std::vector<std::vector<Segment>>& segs_;
  size_t columns_;
  size_t cells_per_column_;
  uint8_t perm_threshold_;
  uint32_t act_threshold_;
  double alpha_fast_;
  double alpha_slow_;
  double ewma_fast_ = 0.0;
  double ewma_slow_ = 0.0;
  std::vector<double> win_;
  size_t win_pos_ = 0;
  size_t win_n_ = 0;
  std::vector<uint8_t> pred_col_;
};

// One evaluation stream: clean transitions label 0, injected anomalies label 1.
// type: 0 = random (novel), 1 = skipped (omission), 2 = repeated (stutter).
std::vector<Sample> RunStream(TemporalMemory& tm, Meter& meter, const std::vector<SDR>& sym,
                              const std::vector<std::vector<int>>& trans, Rng& rng, int warmup,
                               int steps, int type, double p_anom, int& b, SDR& prev_ctx) {
  meter.reset();
  for (int t = 0; t < warmup; ++t) {
    b = PickNext(trans, b, rng);
    tm.step(sym[static_cast<size_t>(b)]);
  }

  std::vector<Sample> out;
  out.reserve(static_cast<size_t>(steps));
  for (int t = 0; t < steps; ++t) {
    const bool anom = rng.unit() < p_anom;
    int c = 0;
    if (!anom) {
      c = PickNext(trans, b, rng);
    } else if (type == 0) {
      c = static_cast<int>(rng.below(static_cast<uint32_t>(sym.size())));
    } else if (type == 1) {
      const int expected = PickNext(trans, b, rng);
      c = PickNext(trans, expected, rng);
    } else {
      c = b;
    }
    b = c;
    const TMStep st = tm.step(sym[static_cast<size_t>(c)]);
    Sample s;
    meter(st, sym[static_cast<size_t>(c)], prev_ctx, s.s);
    s.label = anom ? 1 : 0;
    out.push_back(s);
    prev_ctx = st.active | st.predictive;
  }
  return out;
}

std::vector<double> Column(const std::vector<Sample>& data, int sig) {
  std::vector<double> v;
  v.reserve(data.size());
  for (const Sample& s : data) v.push_back(s.s[sig]);
  return v;
}

std::vector<int> Labels(const std::vector<Sample>& data) {
  std::vector<int> v;
  v.reserve(data.size());
  for (const Sample& s : data) v.push_back(s.label);
  return v;
}

void PrintAucLine(const char* type, const std::vector<Sample>& data) {
  std::printf("nov,auc,type=%s", type);
  for (size_t k = 0; k < kNumSignals; ++k) {
    std::printf(",%s=%.3f", kSignalNames[k], Auc(Column(data, static_cast<int>(k)), Labels(data)));
  }
  std::printf("\n");
}

}  // namespace

int main() {
  // ---------------------------------------------------------------- setup
  const size_t N = 512, K = 20;
  const int symbols = 64;
  Encoder enc(N, K, 9);
  std::vector<SDR> sym;
  sym.reserve(static_cast<size_t>(symbols));
  for (int s = 0; s < symbols; ++s) {
    char name[16];
    std::snprintf(name, sizeof(name), "s%d", s);
    sym.push_back(enc.symbol(name));
  }

  std::vector<std::vector<int>> trans(static_cast<size_t>(symbols));
  Rng rng(23);
  for (size_t i = 0; i < trans.size(); ++i) {
    for (int j = 0; j < 3; ++j) trans[i].push_back(static_cast<int>(rng.below(symbols)));
  }

  TemporalMemoryParams tp;
  tp.column_count = N;
  tp.cells_per_column = 8;
  tp.max_segments_per_cell = 8;
  tp.activation_threshold = 6;
  tp.new_segment_min = 6;
  tp.seed = 11;
  TemporalMemory tm(tp);

  int b = 1;
  const int train_steps = 30000;
  for (int t = 0; t < train_steps; ++t) {
    b = PickNext(trans, b, rng);
    tm.step(sym[static_cast<size_t>(b)]);
  }
  const size_t segments = tm.segment_count();
  tm.set_learning_gain(0.0f);  // evaluation must not change the trained model

  Meter meter(tm, tp, kEwmAlpha, kSlowAlpha);

  // ---------------------------------------------------------------- clean accuracy
  {
    Rng nr(101);
    b = 1;
    const int n = 3000;
    double col_recall = 0.0, burst = 0.0, api_dev = 0.0;
    int step_hits = 0;
    int any_hits = 0;
    SDR prev_ctx(tm.cell_count(), 0);
    for (int t = 0; t < 100; ++t) {
      b = PickNext(trans, b, nr);
      const TMStep st = tm.step(sym[static_cast<size_t>(b)]);
      prev_ctx = st.active | st.predictive;
    }
    for (int t = 0; t < n; ++t) {
      b = PickNext(trans, b, nr);
      const TMStep st = tm.step(sym[static_cast<size_t>(b)]);
      double sig[kNumSignals];
      meter(st, sym[static_cast<size_t>(b)], prev_ctx, sig);
      prev_ctx = st.active | st.predictive;
      col_recall += 1.0 - sig[1];
      burst += sig[0];
      api_dev = std::max(api_dev, std::fabs(tm.anomaly_score() - sig[0]));
      step_hits += (1.0 - sig[1]) > 0.5 ? 1 : 0;
      any_hits += (1.0 - sig[1]) > 0.0 ? 1 : 0;
    }
    std::printf("nov,clean,predict_col_recall=%.3f,predict_any=%.3f,predict_step_acc=%.3f,"
                "burst=%.4f,anomaly_score_dev=%.2e,segments=%zu\n",
                col_recall / static_cast<double>(n), static_cast<double>(any_hits) / static_cast<double>(n),
                static_cast<double>(step_hits) / static_cast<double>(n),
                burst / static_cast<double>(n), api_dev, segments);
  }

  // ---------------------------------------------------------------- ROC per anomaly type
  const char* type_names[3] = {"random", "skip", "stutter"};
  const int stream_steps = 4000;
  const double p_anom = 0.10;
  std::vector<Sample> pooled;
  std::vector<std::vector<Sample>> per_type(3);

  for (int type = 0; type < 3; ++type) {
    tm.reset();
    meter.reset();
    Rng sr(500 + static_cast<uint64_t>(type));
    b = 1;
    SDR prev_ctx(tm.cell_count(), 0);
    per_type[static_cast<size_t>(type)] = RunStream(tm, meter, sym, trans, sr, 200, stream_steps,
                                                    type, p_anom, b, prev_ctx);
    PrintAucLine(type_names[type], per_type[static_cast<size_t>(type)]);
    pooled.insert(pooled.end(), per_type[static_cast<size_t>(type)].begin(),
                  per_type[static_cast<size_t>(type)].end());
  }
  PrintAucLine("pooled", pooled);

  // ---------------------------------------------------------------- control: shuffled labels
  {
    std::vector<int> labels = Labels(pooled);
    Rng sh(777);
    for (size_t i = labels.size(); i > 1; --i) {
      const size_t j = static_cast<size_t>(sh.below(static_cast<uint32_t>(i)));
      std::swap(labels[i - 1], labels[j]);
    }
    std::vector<Sample> shuffled = pooled;
    for (size_t i = 0; i < shuffled.size(); ++i) shuffled[i].label = labels[i];
    std::printf("nov,auc_shuffled_control");
    for (size_t k = 0; k < kNumSignals; ++k) {
      std::printf(",%s=%.3f", kSignalNames[k],
                  Auc(Column(shuffled, static_cast<int>(k)), Labels(shuffled)));
    }
    std::printf("\n");
  }

  // ---------------------------------------------------------------- EWMA alpha sweep
  // Sensitivity of the persistence signal to its time constant. Computed offline
  // from the saved pred_error sequence, so this is a sweep over a derived
  // quantity, not a second training run. Selection on the same data: reported
  // for transparency, the main table keeps the a-priori alpha = 0.2.
  {
    std::printf("nov,alpha_sweep");
    for (double alpha : {0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.80}) {
      std::vector<double> score;
      std::vector<int> label;
      for (int type = 0; type < 3; ++type) {
        double v = 0.0;
        for (const Sample& s : per_type[static_cast<size_t>(type)]) {
          v = (1.0 - alpha) * v + alpha * s.s[1];
          score.push_back(v);
          label.push_back(s.label);
        }
      }
      std::printf(",alpha=%.2f:%.3f", alpha, Auc(score, label));
    }
    std::printf("\n");
  }

  // ---------------------------------------------------------------- pick best signal
  int best_sig = 0;
  double best_mean = -1.0;
  for (size_t k = 0; k < kNumSignals; ++k) {
    double sum = 0.0;
    for (int type = 0; type < 3; ++type) {
      sum += Auc(Column(per_type[static_cast<size_t>(type)], static_cast<int>(k)),
                 Labels(per_type[static_cast<size_t>(type)]));
    }
    const double mean_auc = sum / 3.0;
    std::printf("nov,mean_auc,%s=%.3f\n", kSignalNames[k], mean_auc);
    if (mean_auc > best_mean) {
      best_mean = mean_auc;
      best_sig = static_cast<int>(k);
    }
  }
  std::printf("nov,best_signal=%s,mean_auc=%.3f\n", kSignalNames[best_sig], best_mean);

  // ---------------------------------------------------------------- threshold at 1% FPR
  {
    std::vector<double> clean, pos;
    for (const Sample& s : pooled) {
      if (s.label > 0) {
        pos.push_back(s.s[best_sig]);
      } else {
        clean.push_back(s.s[best_sig]);
      }
    }
    const OpPoint op = ThresholdAtFpr(clean, pos, 0.01);
    std::printf("nov,threshold_at_fpr1=%.4f,tpr=%.3f,signal=%s\n", op.threshold, op.tpr,
                kSignalNames[best_sig]);
    std::printf("nov,threshold_fpr_actual=%.4f,clean_n=%zu,anom_n=%zu\n", op.fpr, clean.size(),
                pos.size());
    for (double target : {0.01, 0.05, 0.10}) {
      const OpPoint q = ThresholdAtFpr(clean, pos, target);
      std::printf("nov,operating_point,target_fpr=%.2f,threshold=%.4f,tpr=%.3f,fpr=%.4f\n",
                  target, q.threshold, q.tpr, q.fpr);
    }
  }

  // ---------------------------------------------------------------- sequential test
  {
    tm.reset();
    meter.reset();
    Rng sr(909);
    b = 1;
    SDR prev_ctx(tm.cell_count(), 0);
    const std::vector<Sample> clean =
        RunStream(tm, meter, sym, trans, sr, 200, 2000, 0, 0.0, b, prev_ctx);
    const std::vector<Sample> corrupt = RunStream(tm, meter, sym, trans, sr, 0, 200, 0, 1.0, b, prev_ctx);

    const size_t bin = 200;
    std::printf("nov,bins=clean:");
    for (size_t i = 0; i < clean.size(); i += bin) {
      double sum = 0.0;
      size_t n = 0;
      for (size_t j = i; j < std::min(i + bin, clean.size()); ++j) {
        sum += clean[j].s[best_sig];
        ++n;
      }
      std::printf("%s%.3f", i ? "," : "", n ? sum / static_cast<double>(n) : 0.0);
    }
    std::printf(";corrupt:");
    for (size_t i = 0; i < corrupt.size(); i += bin) {
      double sum = 0.0;
      size_t n = 0;
      for (size_t j = i; j < std::min(i + bin, corrupt.size()); ++j) {
        sum += corrupt[j].s[best_sig];
        ++n;
      }
      std::printf("%s%.3f", i ? "," : "", n ? sum / static_cast<double>(n) : 0.0);
    }
    std::printf("\n");
  }

  std::printf("nov,peak_rss=%zuMB\n", PeakRssMb());
  return 0;
}