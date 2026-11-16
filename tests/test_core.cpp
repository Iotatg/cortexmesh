#include <cmath>
#include <vector>

#include "bm/encoder.hpp"
#include "bm/region.hpp"
#include "bm/modulator.hpp"
#include "bm/salience.hpp"
#include "bm/sdr.hpp"
#include "bm/spatial_pooler.hpp"
#include "bm/temporal_memory.hpp"
#include "test_harness.hpp"

using namespace bm;

static SDR one_hot(size_t n, size_t i) {
  SDR s(n, 0);
  s.set(static_cast<uint32_t>(i));
  return s;
}

static void test_sdr() {
  SDR a(1024, 40);
  CHECK(a.count() == 40);
  CHECK(a.size() == 1024);
  CHECK(is_sparse_valid(a, 0.03, 0.05));
  CHECK(a.overlap(a) == 40);
  CHECK(a.jaccard(a) == 1.0);

  SDR b = SDR::random(1024, 40, 12345);
  SDR c = SDR::random(1024, 40, 999);
  CHECK(a != b);
  CHECK(b != c);
  CHECK(b.overlap(c) < 40);

  SDR u = b | c;
  CHECK(u.count() == b.count() + c.count() - b.overlap(c));
  SDR inter = b & c;
  CHECK(inter.count() == b.overlap(c));

  SDR d = b;
  d.set(1000);
  CHECK(d.test(1000));
  d.reset(1000);
  CHECK(!d.test(1000));

  uint64_t h1 = b.hash();
  SDR b2 = b;
  CHECK(h1 == b2.hash());
}

static void test_encoder() {
  Encoder enc(2048, 40, 5);
  SDR x = enc.symbol("cat");
  SDR y = enc.symbol("cat");
  SDR z = enc.symbol("dog");
  CHECK(x == y);
  CHECK(x != z);
  CHECK(is_sparse_valid(x, 0.015, 0.025));

  ScalarEncoder se(1024, 32, 0.0, 10.0, 20);
  SDR v1 = se.encode(1.0);
  SDR v2 = se.encode(1.2);
  SDR v9 = se.encode(9.0);
  CHECK(v1.overlap(v2) > v1.overlap(v9));

  SDR g = enc.grid(3.0, 4.0, 4, 64);
  CHECK(g.count() > 0);

  SDR ctx = enc.context({"a", "b", "c"}, 2);
  CHECK(ctx.count() > 0);
}

static void test_spatial_pooler() {
  SpatialPoolerParams p;
  p.input_size = 1024;
  p.column_count = 512;
  p.synapses_per_column = 64;
  p.active_columns = 16;
  SpatialPooler sp(p);

  Rng rng(9);
  SDR in = SDR::random(1024, 40, rng.next());
  SDR cols = sp.activate(in);
  CHECK(cols.count() == 16);
  CHECK(is_sparse_valid(cols, 0.02, 0.05));

  double same = 0.0;
  int trials = 20;
  for (int i = 0; i < trials; ++i) {
    SDR again = sp.activate(in);
    same += static_cast<double>(again.overlap(cols)) / 16.0;
  }
  same /= trials;
  CHECK(same > 0.95);

  double diff = 0.0;
  for (int i = 0; i < 20; ++i) {
    SDR other = SDR::random(1024, 40, rng.next());
    SDR oc = sp.activate(other);
    diff += static_cast<double>(oc.overlap(cols)) / 16.0;
  }
  diff /= 20.0;
  CHECK(diff < same - 0.5);

  for (int i = 0; i < 300; ++i) {
    SDR s = SDR::random(1024, 40, rng.next());
    sp.activate(s);
  }
  CHECK(sp.dead_columns() < 64);
}

static void test_temporal_memory() {
  TemporalMemoryParams p;
  p.column_count = 64;
  p.cells_per_column = 16;
  p.activation_threshold = 1;
  p.min_threshold = 1;
  p.new_segment_min = 1;
  p.permanence_threshold = 10;
  TemporalMemory tm(p);

  size_t n = 32;
  std::vector<SDR> seq;
  for (size_t i = 0; i < n; ++i) seq.push_back(one_hot(64, i));

  for (int epoch = 0; epoch < 30; ++epoch) {
    for (size_t i = 0; i < n; ++i) tm.step(seq[i]);
  }

  size_t correct = 0;
  size_t bursts = 0;
  for (size_t i = 0; i < n; ++i) {
    TMStep st = tm.step(seq[i]);
    if (st.bursting_columns == 0) ++correct;
    bursts += st.bursting_columns;
  }
  double acc = static_cast<double>(correct) / static_cast<double>(n);
  CHECK(acc > 0.9);
  CHECK(bursts == 0);
  CHECK(tm.segment_count() > 0);

  SDR unseen = one_hot(64, 40);
  TMStep st = tm.step(unseen);
  CHECK(st.bursting_columns == 1);
  CHECK(st.predicted_active == 0);
}

static void test_hierarchy_prediction_error() {
  SpatialPoolerParams p;
  p.input_size = 512;
  p.column_count = 1024;
  p.synapses_per_column = 64;
  p.active_columns = 20;
  SpatialPooler sp(p);

  TemporalMemoryParams t;
  t.column_count = 1024;
  t.cells_per_column = 16;
  t.activation_threshold = 2;
  t.permanence_threshold = 10;
  TemporalMemory tm(t);

  Rng rng(21);
  std::vector<SDR> items;
  for (int i = 0; i < 8; ++i) items.push_back(SDR::random(512, 20, rng.next()));

  for (int epoch = 0; epoch < 20; ++epoch) {
    for (int rep = 0; rep < 3; ++rep) {
      for (const SDR& it : items) {
        SDR cols = sp.activate(it);
        tm.step(cols);
      }
    }
  }

  size_t predicted = 0;
  size_t total = 0;
  for (int rep = 0; rep < 3; ++rep) {
    for (const SDR& it : items) {
      SDR cols = sp.activate(it);
      TMStep st = tm.step(cols);
      predicted += st.predicted_active;
      total += st.active.count();
    }
  }
  double acc = total ? static_cast<double>(predicted) / static_cast<double>(total) : 0.0;
  CHECK(acc > 0.5);
}

static void test_region() {
  RegionParams rp;
  rp.input_size = 1024;
  rp.column_count = 1024;
  rp.cells_per_column = 16;
  rp.synapses_per_column = 64;
  rp.active_columns = 20;
  Region region(rp);

  Rng rng(77);
  std::vector<SDR> items;
  for (int i = 0; i < 8; ++i) items.push_back(SDR::random(1024, 20, rng.next()));

  for (int epoch = 0; epoch < 15; ++epoch) {
    for (const SDR& it : items) region.step(it);
  }

  size_t confident = 0;
  for (const SDR& it : items) {
    RegionStep st = region.step(it);
    if (st.bursting_columns == 0) ++confident;
  }
  double acc = static_cast<double>(confident) / static_cast<double>(items.size());
  CHECK(acc > 0.8);
  CHECK(region.segment_count() > 0);
  CHECK(region.prediction_accuracy() > 0.5);

  SDR unseen = SDR::random(1024, 20, rng.next());
  RegionStep st = region.step(unseen);
  CHECK(st.anomaly > 0.5);
}

static void test_focus_speeds_learning() {
  auto train = [](float focus) {
    RegionParams rp;
    rp.input_size = 1024;
    rp.column_count = 1024;
    rp.active_columns = 20;
    Region r(rp);
    r.set_focus(focus);
    Rng rng(31);
    std::vector<SDR> items;
    for (int i = 0; i < 8; ++i) items.push_back(SDR::random(1024, 20, rng.next()));
    size_t correct = 0;
    for (int epoch = 0; epoch < 8; ++epoch) {
      correct = 0;
      for (const SDR& it : items) {
        RegionStep st = r.step(it);
        if (st.bursting_columns == 0) ++correct;
      }
    }
    return static_cast<double>(correct) / static_cast<double>(items.size());
  };
  double normal = train(1.0f);
  double aroused = train(12.0f);
  CHECK(normal > aroused);
}

static void test_modulators() {
  ModulatorSystem mod;
  mod.observe_state(1);
  mod.observe_reward(1.0f);
  CHECK(mod.modulators().rpe > 0.5f);
  CHECK(mod.value() > 0.0f);

  for (int i = 0; i < 200; ++i) {
    mod.observe_state(1);
    mod.observe_reward(1.0f);
  }
  CHECK(std::fabs(mod.modulators().rpe) < 0.2f);

  for (int i = 0; i < 10; ++i) mod.observe_anomaly(1.0f);
  float gain_novel = mod.modulators().gain();
  CHECK(gain_novel > 1.2f);
  CHECK(mod.modulators().na > 1.0f);

  for (int i = 0; i < 500; ++i) {
    mod.tick();
    mod.observe_anomaly(0.0f);
  }
  CHECK(mod.modulators().novelty < 0.2f);
  CHECK(mod.modulators().gain() < gain_novel);
}

static void test_salience() {
  Salience sal;
  for (int i = 0; i < 20; ++i) sal.update(1.0f);
  float high = sal.gain();
  CHECK(high > 1.5f);
  CHECK(sal.learning_rate(0.05f) > 0.05f);

  for (int i = 0; i < 400; ++i) sal.update(0.0f);
  CHECK(sal.gain() < high);
}

int main() {
  test_sdr();
  test_encoder();
  test_spatial_pooler();
  test_temporal_memory();
  test_hierarchy_prediction_error();
  test_region();
  test_focus_speeds_learning();
  test_modulators();
  test_salience();
  return bmtest::summary("core");
}