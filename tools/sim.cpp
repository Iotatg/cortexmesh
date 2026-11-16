#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "bm/encoder.hpp"
#include "bm/region.hpp"
#include "bm/sdr.hpp"

using namespace bm;

namespace {

constexpr int kW = 24;
constexpr int kH = 24;
constexpr int kPatch = 9;
constexpr int kBits = 2048;

int t_global = 0;

struct World {
  int ax = 3;
  int ay = 3;
  int dir = 1;
  int speed = 1;
  bool pass_wall = false;
  bool spin = false;
  int food_dup = 0;
  int noise_bits = 0;
  std::vector<std::pair<int, int>> food;
  Rng rng{1234};

  World() { food = {{10, 10}, {17, 6}, {6, 17}}; }

  bool wall(int x, int y) const { return x < 0 || y < 0 || x >= kW || y >= kH; }

  void step() {
    if (spin) dir = (dir + 1) % 4;
    int dx = dir == 1 ? 1 : dir == 3 ? -1 : 0;
    int dy = dir == 2 ? 1 : dir == 0 ? -1 : 0;
    int nx = ax + dx * speed;
    int ny = ay + dy * speed;
    if (wall(nx, ny)) {
      if (pass_wall) {
        ax = (nx + kW) % kW;
        ay = (ny + kH) % kH;
      } else {
        dir = (dir + 2) % 4;
      }
    } else {
      ax = nx;
      ay = ny;
    }
    ax = std::max(0, std::min(kW - 1, ax));
    ay = std::max(0, std::min(kH - 1, ay));
    for (auto& f : food) {
      if (f.first == ax && f.second == ay) {
        int x = 1 + static_cast<int>(rng.below(static_cast<uint32_t>(kW - 2)));
        int y = 1 + static_cast<int>(rng.below(static_cast<uint32_t>(kH - 2)));
        f = {x, y};
      }
    }
    for (int i = 0; i < food_dup; ++i) {
      int fx = (ax + 2 + i) % kW;
      int fy = (ay + (i % 3) - 1 + kH) % kH;
      food.push_back({fx, fy});
    }
    food_dup = 0;
  }
};

bool occupied(const World& w, int x, int y) {
  if (w.wall(x, y)) return true;
  for (const auto& f : w.food) {
    if (f.first == x && f.second == y) return true;
  }
  return false;
}

SDR observe(const World& w) {
  SDR s(kBits, 0);
  for (int dy = -kPatch / 2; dy <= kPatch / 2; ++dy) {
    for (int dx = -kPatch / 2; dx <= kPatch / 2; ++dx) {
      if (occupied(w, w.ax + dx, w.ay + dy)) {
        s.set(static_cast<uint32_t>((dy + kPatch / 2) * kPatch + (dx + kPatch / 2)));
      }
    }
  }
  int wall_dist = kW;
  for (int d = 0; d < kW; ++d) {
    if (w.wall(w.ax + d, w.ay) || w.wall(w.ax - d, w.ay) || w.wall(w.ax, w.ay + d) ||
        w.wall(w.ax, w.ay - d)) {
      wall_dist = d;
      break;
    }
  }
  s.set(static_cast<uint32_t>(128 + std::min(wall_dist, 8)));
  return s;
}

SDR observe_state(const World& w) {
  static Encoder enc(4096, 20, 21);
  uint32_t id = static_cast<uint32_t>(((w.ay % kH) * kW + (w.ax % kW)) * 4 + (w.dir % 4));
  return enc.symbol("state:" + std::to_string(id));
}

struct Condition {
  const char* name;
  int t0;
  int t1;
  std::function<void(World&, int)> mutate;
};

RegionParams patch_params(int act_thresh, int cells, bool use_pred) {
  RegionParams p;
  p.input_size = kBits;
  p.column_count = 4096;
  p.cells_per_column = static_cast<size_t>(cells);
  p.synapses_per_column = 64;
  p.active_columns = 20;
  p.use_predictive_in_prev = use_pred;
  p.activation_threshold = static_cast<uint32_t>(act_thresh);
  return p;
}

void train_clean(Region& patch_ai, Region& state_ai, int epochs, int steps) {
  for (int e = 0; e < epochs; ++e) {
    World w;
    t_global = 0;
    for (int t = 0; t < steps; ++t) {
      patch_ai.step(observe(w));
      state_ai.step(observe_state(w));
      w.step();
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  int epochs = argc > 1 ? std::atoi(argv[1]) : 15;
  int steps = argc > 2 ? std::atoi(argv[2]) : 300;
  bool use_pred = argc > 3 ? std::atoi(argv[3]) != 0 : true;
  int act_thresh = argc > 4 ? std::atoi(argv[4]) : 2;
  int cells = argc > 5 ? std::atoi(argv[5]) : 16;

  std::vector<Condition> conditions = {
      {"clean", 0, steps, [](World&, int) {}},
      {"teleport", 25, 125, [](World& w, int t) {
         if (t % 25 == 0) {
           w.ax = 1 + static_cast<int>(w.rng.below(static_cast<uint32_t>(kW - 2)));
           w.ay = 1 + static_cast<int>(w.rng.below(static_cast<uint32_t>(kH - 2)));
         }
       }},
      {"pass_wall", 40, 140, [](World& w, int t) { w.pass_wall = (t >= 40 && t < 140); }},
      {"speed2", 40, 140, [](World& w, int t) { w.speed = (t >= 40 && t < 140) ? 2 : 1; }},
      {"speed3", 40, 140, [](World& w, int t) { w.speed = (t >= 40 && t < 140) ? 3 : 1; }},
      {"spin", 40, 140, [](World& w, int t) { w.spin = (t >= 40 && t < 140); }},
      {"food_dup", 60, 160, [](World& w, int t) {
         if (t == 60) w.food_dup = 8;
       }},
      {"noise1", 0, steps, [](World& w, int) { w.noise_bits = 1; }},
      {"noise3", 0, steps, [](World& w, int) { w.noise_bits = 3; }},
  };

  std::printf("condition,patch_burst,patch_run,state_burst,state_run,pred_acc,verdict\n");
  std::vector<double> patch_rate;
  std::vector<double> state_rate;

  for (const auto& c : conditions) {
    Region patch_ai(patch_params(act_thresh, cells, use_pred));
    RegionParams sp = patch_params(act_thresh, cells, use_pred);
    sp.input_size = 4096;
    Region state_ai(sp);
    train_clean(patch_ai, state_ai, epochs, steps);

    World w;
    t_global = 0;
    size_t bursts = 0, counted = 0, run = 0, runs = 0, run_total = 0, run_max = 0;
    size_t sbursts = 0, srun = 0, sruns = 0, srun_total = 0, srun_max = 0;
    double acc = 0.0;

    for (int t = 0; t < steps; ++t) {
      c.mutate(w, t);
      SDR obs = observe(w);
      for (int b = 0; b < w.noise_bits; ++b) {
        uint32_t bit = static_cast<uint32_t>(w.rng.below(140));
        if (obs.test(bit)) obs.reset(bit);
        else obs.set(bit);
      }
      bool in_window = t >= c.t0 && t < c.t1;
      RegionStep st = patch_ai.step(obs);
      RegionStep sst = state_ai.step(observe_state(w));
      if (in_window) {
        ++counted;
        acc += patch_ai.prediction_accuracy();
        if (st.bursting_columns > 0) {
          ++bursts;
          ++run;
          run_max = std::max(run_max, run);
        } else if (run > 0) {
          ++runs;
          run_total += run;
          run = 0;
        }
        if (sst.bursting_columns > 0) {
          ++sbursts;
          ++srun;
          srun_max = std::max(srun_max, srun);
        } else if (srun > 0) {
          ++sruns;
          srun_total += srun;
          srun = 0;
        }
      }
      w.step();
    }
    if (counted == 0) counted = 1;
    if (run > 0) { ++runs; run_total += run; }
    if (srun > 0) { ++sruns; srun_total += srun; }

    double rate = static_cast<double>(bursts) / counted;
    double srate = static_cast<double>(sbursts) / counted;
    double mean_run = runs ? static_cast<double>(run_total) / runs : 0.0;
    double smean = sruns ? static_cast<double>(srun_total) / sruns : 0.0;
    patch_rate.push_back(rate);
    state_rate.push_back(srate);

    const double threshold = 0.10;
    std::printf("%s,%.3f,%.2f,%.3f,%.2f,%.3f,%s\n", c.name, rate, mean_run, srate, smean,
                acc / counted, srate > threshold ? "ANOMALY" : "consistent");
  }

  double clean = patch_rate[0];
  double fp = state_rate[0];
  double detection = 1.0;
  size_t flagged = 0, glitch_total = 0;
  for (size_t i = 1; i < conditions.size(); ++i) {
    std::string name = conditions[i].name;
    if (name.rfind("noise", 0) == 0) {
      fp = std::max(fp, state_rate[i]);
    } else {
      ++glitch_total;
      detection = std::min(detection, state_rate[i]);
      if (state_rate[i] > 0.10) ++flagged;
    }
  }
  std::printf("summary,clean=%.3f,false_positive=%.3f,detection_floor=%.3f,margin=%.3f,"
              "detected=%zu/%zu\n",
              clean, fp, detection, detection - fp, flagged, glitch_total);
  return 0;
}