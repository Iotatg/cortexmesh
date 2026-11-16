#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "bm/dual_controller.hpp"
#include "bm/hippocampus.hpp"
#include "bm/interoception.hpp"
#include "bm/modulator.hpp"
#include "bm/region.hpp"
#include "bm/salience.hpp"
#include "bm/valuation.hpp"

using namespace bm;

static SDR code(size_t n, size_t k, size_t i) {
  Rng r(0xABCDull + i * 7919ull);
  return SDR::random(n, k, r.next());
}

static double seq_accuracy(Region& r, const std::vector<SDR>& items, int epochs, bool train) {
  size_t ok = 0;
  size_t total = 0;
  for (int e = 0; e < epochs; ++e) {
    ok = 0;
    total = 0;
    for (const SDR& it : items) {
      RegionStep st = r.step(it);
      if (train || e == epochs - 1) {
        if (st.bursting_columns == 0) ++ok;
        ++total;
      }
    }
  }
  return total ? static_cast<double>(ok) / static_cast<double>(total) : 0.0;
}

static void exp_capacity() {
  std::printf("items,columns,active,epochs,accuracy\n");
  for (size_t items_n : {4, 8, 16, 32, 64}) {
    for (size_t mult : {8, 16, 32, 64}) {
      RegionParams p;
      p.input_size = 1024;
      p.column_count = items_n * mult;
      p.cells_per_column = 16;
      p.synapses_per_column = 64;
      p.active_columns = 20;
      Region r(p);
      std::vector<SDR> items;
      for (size_t i = 0; i < items_n; ++i) items.push_back(code(1024, 20, i));
      double acc = seq_accuracy(r, items, 25, true);
      std::printf("%zu,%zu,20,25,%.3f\n", items_n, p.column_count, acc);
    }
  }
}

static void exp_eligibility() {
  std::printf("trace_decay,delay,diff\n");
  for (float decay : {0.0f, 0.5f, 0.7f, 0.85f, 0.95f, 0.99f}) {
    for (int delay : {1, 5, 20}) {
      Valuation v(2048, 0.05f, decay);
      SDR a = code(2048, 20, 1);
      SDR b = code(2048, 20, 2);
      for (int rep = 0; rep < 300; ++rep) {
        v.present(a);
        for (int d = 0; d < delay; ++d) v.tick();
        v.reward(1.0f);
        v.tick();
        v.present(b);
        for (int d = 0; d < delay; ++d) v.tick();
        v.reward(0.0f);
        v.tick();
      }
      float diff = v.predict(a) - v.predict(b);
      std::printf("%.2f,%d,%.3f\n", static_cast<double>(decay), delay, static_cast<double>(diff));
    }
  }
}

static void exp_replay() {
  std::printf("order,acc_before,acc_after\n");
  for (int order = 0; order < 4; ++order) {
    RegionParams p;
    p.input_size = 1024;
    p.column_count = 2048;
    p.cells_per_column = 16;
    p.synapses_per_column = 64;
    p.active_columns = 20;
    Region r(p);

    std::vector<SDR> A, B;
    for (int i = 0; i < 8; ++i) A.push_back(code(1024, 20, i + 100));
    for (int i = 0; i < 8; ++i) B.push_back(code(1024, 20, i + 200));

    for (int e = 0; e < 12; ++e) {
      for (const SDR& x : A) r.step(x);
    }
    double before = seq_accuracy(r, A, 1, false);

    for (int e = 0; e < 12; ++e) {
      for (const SDR& x : B) r.step(x);
    }

    std::vector<SDR> replay = A;
    if (order == 1) std::reverse(replay.begin(), replay.end());
    if (order == 2) {
      Rng rng(5);
      for (size_t i = replay.size(); i > 1; --i) std::swap(replay[i - 1], replay[rng.below(static_cast<uint32_t>(i))]);
    }
    if (order != 3) {
      for (int e = 0; e < 4; ++e) {
        for (const SDR& x : replay) r.step(x);
      }
    }
    double after = seq_accuracy(r, A, 1, false);
    const char* name = order == 0 ? "forward" : (order == 1 ? "reverse" : (order == 2 ? "random" : "none"));
    std::printf("%s,%.3f,%.3f\n", name, before, after);
  }
}

static void exp_replay_scale() {
  std::printf("items,mode,acc\n");
  const char* names[] = {"none", "end5", "end10", "interleaved"};
  for (size_t items_n : {8, 16, 32}) {
    for (int mode = 0; mode < 4; ++mode) {
      RegionParams p;
      p.input_size = 1024;
      p.column_count = items_n * 32;
      p.cells_per_column = 16;
      p.synapses_per_column = 64;
      p.active_columns = 20;
      Region r(p);
      std::vector<SDR> A, B;
      for (size_t i = 0; i < items_n; ++i) A.push_back(code(1024, 20, i + 100));
      for (size_t i = 0; i < items_n; ++i) B.push_back(code(1024, 20, i + 200));
      for (int e = 0; e < 15; ++e) {
        for (const SDR& x : A) r.step(x);
      }
      if (mode == 3) {
        size_t step = 0;
        for (int e = 0; e < 15; ++e) {
          for (const SDR& x : B) {
            r.step(x);
            if (++step % 4 == 0) r.step(A[(step / 4) % A.size()]);
          }
        }
      } else {
        for (int e = 0; e < 15; ++e) {
          for (const SDR& x : B) r.step(x);
        }
        if (mode == 1) {
          for (int e = 0; e < 5; ++e) {
            for (const SDR& x : A) r.step(x);
          }
        } else if (mode == 2) {
          for (int e = 0; e < 10; ++e) {
            for (const SDR& x : A) r.step(x);
          }
        }
      }
      size_t ok = 0;
      for (const SDR& x : A) {
        RegionStep st = r.step(x);
        if (st.bursting_columns == 0) ++ok;
      }
      std::printf("%zu,%s,%.3f\n", items_n, names[mode],
                  static_cast<double>(ok) / static_cast<double>(A.size()));
    }
  }
}

static void exp_interoception() {
  std::printf("model,mae\n");
  Interoception io(512, 64);
  std::vector<SDR> states;
  for (int i = 0; i < 6; ++i) states.push_back(code(1024, 20, i + 600));
  auto reward_of = [](int s, bool calm) { (void)s; return calm ? 1.0f : 0.0f; };

  Valuation v_state(4096, 0.05f, 0.0f);
  Valuation v_body(4096, 0.05f, 0.0f);
  Rng rng(13);
  for (int rep = 0; rep < 400; ++rep) {
    int s = static_cast<int>(rng.below(static_cast<uint32_t>(states.size())));
    bool calm = rng.below(2) == 0;
    BodyState b = calm ? BodyState{62, 80, 12, 0.1f} : BodyState{120, 20, 24, 0.9f};
    float r = reward_of(s, calm);

    SDR sa = states[s];
    v_state.present(sa);
    v_state.reward(r);
    v_state.tick();

    SDR sb(4096);
    for (uint32_t idx : states[s].indices()) sb.set(idx);
    for (uint32_t idx : io.encode(b).indices()) sb.set(static_cast<uint32_t>(2048 + idx));
    v_body.present(sb);
    v_body.reward(r);
    v_body.tick();
  }

  double e1 = 0.0, e2 = 0.0;
  int n = 0;
  for (int s = 0; s < 6; ++s) {
    for (int calm = 0; calm < 2; ++calm) {
      bool c = calm == 1;
      BodyState b = c ? BodyState{62, 80, 12, 0.1f} : BodyState{120, 20, 24, 0.9f};
      float r = reward_of(s, c);
      e1 += std::fabs(v_state.predict(states[s]) - r);
      SDR sb(4096);
      for (uint32_t idx : states[s].indices()) sb.set(idx);
      for (uint32_t idx : io.encode(b).indices()) sb.set(static_cast<uint32_t>(2048 + idx));
      e2 += std::fabs(v_body.predict(sb) - r);
      ++n;
    }
  }
  std::printf("state_only,%.3f\n", e1 / n);
  std::printf("state_plus_body,%.3f\n", e2 / n);
}

static void exp_ceiling() {
  std::printf("items,columns,epochs,accuracy\n");
  for (size_t items_n : {64, 128, 256}) {
    for (size_t mult : {32, 64}) {
      RegionParams p;
      p.input_size = 1024;
      p.column_count = items_n * mult;
      p.cells_per_column = 16;
      p.synapses_per_column = 64;
      p.active_columns = 20;
      Region r(p);
      std::vector<SDR> items;
      for (size_t i = 0; i < items_n; ++i) items.push_back(code(1024, 20, i));
      size_t ok = 0;
      for (int e = 0; e < 15; ++e) {
        ok = 0;
        for (const SDR& x : items) {
          RegionStep st = r.step(x);
          if (st.bursting_columns == 0) ++ok;
        }
      }
      std::printf("%zu,%zu,15,%.3f\n", items_n, p.column_count,
                  static_cast<double>(ok) / static_cast<double>(items.size()));
    }
  }
}

static void exp_prune() {
  std::printf("items,columns,prune,acc\n");
  for (size_t items_n : {8, 16}) {
    for (size_t mult : {8, 16}) {
      for (int do_prune = 0; do_prune < 2; ++do_prune) {
        SpatialPoolerParams sp;
        sp.input_size = 1024;
        sp.column_count = items_n * mult;
        sp.synapses_per_column = 64;
        sp.active_columns = 20;
        SpatialPooler pool(sp);
        TemporalMemoryParams t;
        t.column_count = items_n * mult;
        t.cells_per_column = 16;
        t.activation_threshold = 2;
        t.permanence_threshold = 10;
        TemporalMemory tm(t);
        std::vector<SDR> items;
        for (size_t i = 0; i < items_n; ++i) items.push_back(code(1024, 20, i));
        for (int e = 0; e < 25; ++e) {
          for (const SDR& x : items) tm.step(pool.activate(x));
        }
        size_t removed = do_prune ? pool.prune(2.0f) : 0;
        if (do_prune) {
          for (int e = 0; e < 10; ++e) {
            for (const SDR& x : items) tm.step(pool.activate(x));
          }
        }
        size_t ok = 0;
        for (const SDR& x : items) {
          TMStep st = tm.step(pool.activate(x));
          if (st.bursting_columns == 0) ++ok;
        }
        std::printf("%zu,%zu,%d,%.3f,%zu\n", items_n, sp.column_count, do_prune,
                    static_cast<double>(ok) / static_cast<double>(items.size()), removed);
      }
    }
  }
}

static SDR corrupt(const SDR& in, size_t flips, Rng& rng) {
  SDR out = in;
  for (size_t f = 0; f < flips; ++f) {
    uint32_t bit = static_cast<uint32_t>(rng.below(1024));
    if (out.test(bit)) out.reset(bit);
    else out.set(bit);
  }
  return out;
}

static void exp_feedback() {
  std::printf("gain,corruption,noburst,recovery\n");
  RegionParams base1;
  base1.input_size = 1024;
  base1.column_count = 1024;
  base1.cells_per_column = 16;
  base1.synapses_per_column = 64;
  base1.active_columns = 20;
  RegionParams p2;
  p2.input_size = 1024;
  p2.column_count = 512;
  p2.cells_per_column = 16;
  p2.synapses_per_column = 64;
  p2.active_columns = 12;

  std::vector<SDR> items;
  for (size_t i = 0; i < 16; ++i) items.push_back(code(1024, 20, i));

  for (double gain : {0.0, 0.5, 1.0, 2.0, 4.0}) {
    RegionParams p1 = base1;
    p1.feedback_gain = gain;
    Region l1(p1);
    Region l2(p2);
    for (int e = 0; e < 25; ++e) {
      for (const SDR& x : items) {
        RegionStep s1 = l1.step(x);
        l2.step(s1.columns);
      }
    }
    std::vector<SDR> clean_cols(items.size());
    for (size_t i = 0; i < items.size(); ++i) clean_cols[i] = l1.step(items[i]).columns;

    for (size_t flips : {0, 5, 10, 20}) {
      Rng rng(21);
      size_t noburst = 0;
      double rec = 0.0;
      bool gate = true;
      for (size_t i = 0; i < items.size(); ++i) {
        size_t prev = (i + items.size() - 1) % items.size();
        l2.step(clean_cols[prev]);
        SDR pred = (gain > 0.0 && gate) ? l2.predicted_columns() : SDR(1024, 0);
        SDR noisy = corrupt(items[i], flips, rng);
        RegionStep st = l1.step(noisy, pred);
        gate = st.bursting_columns > 0;
        if (st.bursting_columns == 0) ++noburst;
        rec += static_cast<double>(st.columns.overlap(clean_cols[i])) /
               static_cast<double>(clean_cols[i].count());
      }
      std::printf("%.1f,%zu,%.3f,%.3f\n", gain, flips,
                  static_cast<double>(noburst) / items.size(), rec / items.size());
    }
  }
}

static void exp_var_order() {
  std::printf("case,input,burst_rate\n");
  Encoder enc(1024, 20, 5);
  SDR x = enc.symbol("x"), a = enc.symbol("a"), b = enc.symbol("b"), c = enc.symbol("c");
  SDR pp = enc.symbol("p"), q = enc.symbol("q");

  auto measure = [&](bool context, uint64_t seed) {
    RegionParams p;
    p.input_size = 1024;
    p.column_count = 512;
    p.cells_per_column = 16;
    p.synapses_per_column = 64;
    p.active_columns = 20;
    p.seed = seed;
    Region r(p);
    for (int e = 0; e < 30; ++e) {
      for (int rep = 0; rep < 2; ++rep) {
        std::vector<SDR> s1 = context ? std::vector<SDR>{pp, x, a} : std::vector<SDR>{x, a};
        for (const SDR& s : s1) r.step(s);
        std::vector<SDR> s2 = context ? std::vector<SDR>{q, x, b} : std::vector<SDR>{x, b};
        for (const SDR& s : s2) r.step(s);
      }
    }
    auto first_burst = [&](const std::vector<SDR>& seq, const SDR& target) {
      for (const SDR& s : seq) {
        RegionStep st = r.step(s);
        if (s == target) return st.bursting_columns > 0 ? 1.0 : 0.0;
      }
      return 0.0;
    };
    std::vector<SDR> seen = context ? std::vector<SDR>{pp, x, a} : std::vector<SDR>{x, a};
    std::vector<SDR> novel = context ? std::vector<SDR>{pp, x, b} : std::vector<SDR>{x, c};
    double seen_b = first_burst(seen, a);
    double novel_b = first_burst(novel, context ? b : c);
    return std::pair<double, double>{seen_b, novel_b};
  };

  for (int context = 0; context < 2; ++context) {
    double seen = 0.0, novel = 0.0;
    int seeds = 5;
    for (int s = 0; s < seeds; ++s) {
      auto r = measure(context != 0, static_cast<uint64_t>(s + 1));
      seen += r.first;
      novel += r.second;
    }
    std::printf("%s,seen,%.3f\n", context ? "context" : "ambiguous", seen / seeds);
    std::printf("%s,novel,%.3f\n", context ? "context" : "ambiguous", novel / seeds);
  }
}

static void exp_hierarchy() {
  std::printf("level,acc\n");
  RegionParams p1;
  p1.input_size = 1024;
  p1.column_count = 1024;
  p1.cells_per_column = 16;
  p1.synapses_per_column = 64;
  p1.active_columns = 20;
  Region l1(p1);
  RegionParams p2;
  p2.input_size = 1024;
  p2.column_count = 512;
  p2.cells_per_column = 16;
  p2.synapses_per_column = 64;
  p2.active_columns = 12;
  Region l2(p2);

  std::vector<SDR> items;
  for (size_t i = 0; i < 16; ++i) items.push_back(code(1024, 20, i));
  for (int e = 0; e < 20; ++e) {
    for (const SDR& x : items) {
      RegionStep s1 = l1.step(x);
      l2.step(s1.columns);
    }
  }
  size_t ok1 = 0, ok2 = 0;
  for (const SDR& x : items) {
    RegionStep s1 = l1.step(x);
    if (s1.bursting_columns == 0) ++ok1;
    RegionStep s2 = l2.step(s1.columns);
    if (s2.bursting_columns == 0) ++ok2;
  }
  std::printf("l1,%.3f\n", static_cast<double>(ok1) / items.size());
  std::printf("l2,%.3f\n", static_cast<double>(ok2) / items.size());
}

static void exp_anomaly_roc() {
  std::printf("corruption,mean_anomaly,burst_rate\n");
  RegionParams p;
  p.input_size = 1024;
  p.column_count = 2048;
  p.cells_per_column = 16;
  p.synapses_per_column = 64;
  p.active_columns = 20;
  Region r(p);
  std::vector<SDR> items;
  for (int i = 0; i < 16; ++i) items.push_back(code(1024, 20, i + 300));
  for (int e = 0; e < 12; ++e) {
    for (const SDR& x : items) r.step(x);
  }
  Rng rng(11);
  for (size_t flips : {0, 2, 5, 10, 20, 40, 80}) {
    double sum = 0.0;
    size_t bursts = 0;
    int trials = 16;
    for (const SDR& x : items) {
      SDR noisy = x;
      for (size_t f = 0; f < flips; ++f) {
        uint32_t bit = static_cast<uint32_t>(rng.below(1024));
        if (noisy.test(bit)) noisy.reset(bit);
        else noisy.set(bit);
      }
      RegionStep st = r.step(noisy);
      sum += st.anomaly;
      if (st.bursting_columns > 0) ++bursts;
    }
    std::printf("%zu,%.3f,%.3f\n", flips, sum / trials, static_cast<double>(bursts) / trials);
  }
}

static void exp_action_sensitivity() {
  std::printf("code_size,code_k,gap_untrained\n");
  for (size_t n : {2048, 8192, 32768}) {
    for (size_t k : {8, 16, 32}) {
      DualControllerParams p;
      p.state_size = 512;
      p.action_size = 512;
      p.code_size = n;
      p.code_k = k;
      DualController dc(p);
      SDR sA = code(512, 20, 1);
      SDR sB = code(512, 20, 2);
      SDR aX = code(512, 20, 3);
      SDR aY = code(512, 20, 4);
      for (int i = 0; i < 150; ++i) dc.learn_model_free(sA, aX, 1.0f);
      for (int i = 0; i < 150; ++i) dc.learn_model_free(sB, aY, 0.2f);
      float gap = dc.model_free_value(sA, aX) - dc.model_free_value(sA, aY);
      std::printf("%zu,%zu,%.3f\n", n, k, gap);
    }
  }
}

static void exp_focus() {
  std::printf("focus,acc\n");
  for (float g : {0.2f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f}) {
    RegionParams p;
    p.input_size = 1024;
    p.column_count = 2048;
    p.cells_per_column = 16;
    p.synapses_per_column = 64;
    p.active_columns = 20;
    Region r(p);
    r.set_focus(g);
    std::vector<SDR> items;
    for (int i = 0; i < 12; ++i) items.push_back(code(1024, 20, i + 400));
    double acc = seq_accuracy(r, items, 4, true);
    std::printf("%.1f,%.3f\n", static_cast<double>(g), acc);
  }
}

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "capacity";
  auto t0 = std::chrono::steady_clock::now();
  if (cmd == "capacity") exp_capacity();
  else if (cmd == "eligibility") exp_eligibility();
  else if (cmd == "replay") exp_replay();
  else if (cmd == "anomaly") exp_anomaly_roc();
  else if (cmd == "action") exp_action_sensitivity();
  else if (cmd == "focus") exp_focus();
  else if (cmd == "replay_scale") exp_replay_scale();
  else if (cmd == "intero") exp_interoception();
  else if (cmd == "ceiling") exp_ceiling();
  else if (cmd == "prune") exp_prune();
  else if (cmd == "hierarchy") exp_hierarchy();
  else if (cmd == "feedback") exp_feedback();
  else if (cmd == "var_order") exp_var_order();
  else std::printf("unknown experiment: %s\n", cmd.c_str());
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "%s took %lld ms\n", cmd.c_str(), static_cast<long long>(ms));
  return 0;
}