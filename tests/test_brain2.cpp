#include <cmath>
#include <vector>

#include "bm/dual_controller.hpp"
#include "bm/encoder.hpp"
#include "bm/region.hpp"
#include "bm/spatial_pooler.hpp"
#include "bm/temporal_memory.hpp"
#include "bm/hippocampus.hpp"
#include "bm/interoception.hpp"
#include "bm/sdr.hpp"
#include "bm/valuation.hpp"
#include "bm/words.hpp"
#include "bm/assoc.hpp"
#include "bm/speech.hpp"
#include "test_harness.hpp"

using namespace bm;

static SDR code(size_t n, size_t k, size_t i) {
  Rng r(0x51ED + i * 7919);
  return SDR::random(n, k, r.next());
}

static void test_valuation() {
  Valuation v(1024, 0.05f, 0.85f);
  SDR a = code(1024, 20, 1);
  SDR b = code(1024, 20, 2);

  for (int i = 0; i < 300; ++i) {
    v.present(a);
    v.tick();
    v.tick();
    v.tick();
    v.reward(1.0f);
    v.tick();
  }
  for (int i = 0; i < 300; ++i) {
    v.present(b);
    v.tick();
    v.tick();
    v.tick();
    v.reward(0.0f);
    v.tick();
  }

  CHECK(v.predict(a) > 0.5f);
  CHECK(v.predict(b) < 0.3f);
  CHECK(v.predict(a) - v.predict(b) > 0.4f);
  CHECK(std::fabs(v.rpe()) < 0.5f);
}

static void test_hippocampus() {
  Hippocampus hp(1024, 2048, 30, 16, 3);
  std::vector<SDR> items;
  for (int i = 0; i < 8; ++i) items.push_back(code(1024, 20, i + 10));
  for (const SDR& it : items) hp.store(it);
  CHECK(hp.stored() == 8);

  SDR d0 = hp.dentate(items[0]);
  SDR d1 = hp.dentate(items[1]);
  CHECK(d0.count() == 30);
  CHECK(d1.count() == 30);
  CHECK(d0.overlap(d1) < 30);
  CHECK(d0.overlap(d0) == 30);

  SDR cue = items[0];
  auto cue_idx = cue.indices();
  for (size_t i = 0; i < cue_idx.size() && i < 6; ++i) cue.reset(cue_idx[i]);
  SDR done = hp.complete(cue, 0.5f);
  CHECK(done.count() == 30);
  CHECK(done.overlap(d0) > 25);
  CHECK(hp.mismatch(items[0], items[1]) > 0.8f);
}

static void test_interoception() {
  Interoception io(512, 64);
  BodyState calm{62, 80, 12, 0.1f};
  BodyState stressed{120, 20, 24, 0.9f};

  SDR c = io.encode(calm);
  SDR s = io.encode(stressed);
  CHECK(c.count() > 0);
  CHECK(s.count() > 0);
  CHECK(c != s);
  CHECK(io.arousal(stressed) > io.arousal(calm));
  CHECK(io.valence(calm) > io.valence(stressed));
  CHECK(io.arousal(calm) >= 0.0f && io.arousal(calm) <= 1.0f);
}

static void test_dual_controller() {
  DualControllerParams p;
  p.state_size = 512;
  p.action_size = 512;
  p.code_size = 2048;
  p.code_k = 24;
  DualController dc(p);

  SDR sA = code(512, 20, 1);
  SDR sB = code(512, 20, 2);
  SDR actX = code(512, 20, 3);
  SDR actY = code(512, 20, 4);

  for (int i = 0; i < 200; ++i) dc.learn_model_free(sA, actX, 1.0f);
  for (int i = 0; i < 200; ++i) dc.learn_model_free(sB, actY, 0.2f);
  for (int i = 0; i < 200; ++i) dc.learn_model_free(sB, actX, 0.0f);
  for (int i = 0; i < 20; ++i) dc.tick();

  float ax = dc.model_free_value(sA, actX);
  float by = dc.model_free_value(sB, actY);
  CHECK(ax > by);

  Decision da = dc.decide(sA, {actX, actY}, 0.0f);
  CHECK(da.action == actX);
  CHECK(da.confidence > 0.0f);

  Decision db = dc.decide(sB, {actX, actY}, 0.0f);
  CHECK(db.action == actY);

  dc.learn_model_based(sB, actY, sA, 1.0f);
  float mb = dc.model_based_value(sB, actY);
  CHECK(mb > 0.0f);
}

static void test_sp_prune() {
  SpatialPoolerParams sp;
  sp.input_size = 1024;
  sp.column_count = 512;
  sp.synapses_per_column = 64;
  sp.active_columns = 20;
  SpatialPooler pool(sp);
  TemporalMemoryParams tp;
  tp.column_count = 512;
  tp.cells_per_column = 16;
  tp.activation_threshold = 2;
  tp.permanence_threshold = 10;
  TemporalMemory tm(tp);
  std::vector<SDR> items;
  for (size_t i = 0; i < 16; ++i) items.push_back(code(1024, 20, i));
  for (int e = 0; e < 25; ++e) {
    for (const SDR& x : items) tm.step(pool.activate(x));
  }
  size_t before = 0;
  for (const SDR& x : items) {
    TMStep st = tm.step(pool.activate(x));
    if (st.bursting_columns == 0) ++before;
  }
  size_t removed = pool.prune(2.0f);
  CHECK(removed > 0);
  size_t pruned = 0;
  for (const SDR& x : items) {
    TMStep st = tm.step(pool.activate(x));
    if (st.bursting_columns == 0) ++pruned;
  }
  CHECK(pruned == before);
  for (int e = 0; e < 5; ++e) {
    for (const SDR& x : items) tm.step(pool.activate(x));
  }
  size_t after = 0;
  for (const SDR& x : items) {
    TMStep st = tm.step(pool.activate(x));
    if (st.bursting_columns == 0) ++after;
  }
  CHECK(after >= 14);
}

static void test_intero_value() {
  Interoception io(512, 64);
  std::vector<SDR> states;
  for (int i = 0; i < 6; ++i) states.push_back(code(1024, 20, i + 600));
  auto body_of = [](bool calm) {
    return calm ? BodyState{62, 80, 12, 0.1f} : BodyState{120, 20, 24, 0.9f};
  };
  Valuation v_state(4096, 0.05f, 0.0f);
  Valuation v_body(4096, 0.05f, 0.0f);
  Rng rng(13);
  for (int rep = 0; rep < 600; ++rep) {
    int s = static_cast<int>(rng.below(static_cast<uint32_t>(states.size())));
    bool calm = rng.below(2) == 0;
    float r = calm ? 1.0f : 0.0f;
    BodyState b = body_of(calm);
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
  for (int s = 0; s < 6; ++s) {
    for (int calm = 0; calm < 2; ++calm) {
      bool c = calm == 1;
      float r = c ? 1.0f : 0.0f;
      BodyState b = body_of(c);
      e1 += std::fabs(v_state.predict(states[s]) - r);
      SDR sb(4096);
      for (uint32_t idx : states[s].indices()) sb.set(idx);
      for (uint32_t idx : io.encode(b).indices()) sb.set(static_cast<uint32_t>(2048 + idx));
      e2 += std::fabs(v_body.predict(sb) - r);
    }
  }
  double mae_state = e1 / 12.0;
  double mae_body = e2 / 12.0;
  CHECK(mae_state > 0.35);
  CHECK(mae_body < 0.15);
}

static void test_feedback_api() {
  RegionParams p;
  p.input_size = 1024;
  p.column_count = 512;
  p.cells_per_column = 16;
  p.synapses_per_column = 64;
  p.active_columns = 20;
  Region r(p);
  std::vector<SDR> items;
  for (size_t i = 0; i < 16; ++i) items.push_back(code(1024, 20, i));
  for (int e = 0; e < 25; ++e) {
    for (const SDR& x : items) r.step(x);
  }
  SDR prev = r.step(items[0]).columns;
  SDR pred = r.predicted_columns();
  CHECK(pred.size() == p.column_count);
  CHECK(pred.count() > 0);
  CHECK(pred.count() <= p.active_columns + 8);
  RegionStep st = r.step(items[1], pred);
  CHECK(st.columns.count() == p.active_columns);
  CHECK(st.anomaly >= 0.0);
  (void)prev;
}

static void test_var_order() {
  Encoder enc(1024, 20, 5);
  SDR x = enc.symbol("x"), a = enc.symbol("a"), b = enc.symbol("b"), c = enc.symbol("c");
  RegionParams p;
  p.input_size = 1024;
  p.column_count = 512;
  p.cells_per_column = 16;
  p.synapses_per_column = 64;
  p.active_columns = 20;
  Region r(p);
  for (int e = 0; e < 30; ++e) {
    for (int rep = 0; rep < 2; ++rep) {
      r.step(x);
      r.step(a);
      r.step(x);
      r.step(b);
    }
  }
  r.step(x);
  size_t seen_burst = 0;
  for (int t = 0; t < 5; ++t) {
    RegionStep s1 = r.step(x);
    (void)s1;
    RegionStep s2 = r.step(a);
    if (s2.bursting_columns > 0) ++seen_burst;
  }
  RegionStep n1 = r.step(x);
  (void)n1;
  RegionStep n2 = r.step(c);
  CHECK(seen_burst <= 1);
  CHECK(n2.bursting_columns > 0);
}


static void test_words_assoc() {
  WordEncoder we(512, 24, 11);
  SDR h = we.encode("bhookh");
  SDR ha = we.encode("bhookha");
  SDR paani = we.encode("paani");
  CHECK(h.count() == 24);
  CHECK(we.encode("bhookh") == h);
  CHECK(h.overlap(ha) > h.overlap(paani));
  CHECK(we.entry("bhookh") != nullptr);

  AssociativeMemory am(512, 512, 24, 0.1f);
  for (int e = 0; e < 60; ++e) {
    am.learn(we.encode("main"), we.encode("bhookha"));
    am.learn(we.encode("mujhe"), we.encode("paani"));
  }
  SDR r1 = am.recall(we.encode("main"));
  SDR r2 = am.recall(we.encode("mujhe"));
  CHECK(we.encode("bhookha").overlap(r1) > we.encode("paani").overlap(r1));
  CHECK(we.encode("paani").overlap(r2) > we.encode("bhookha").overlap(r2));
}


static void test_speech_phase() {
  SpeechModel sm(1024, 24, 31);
  sm.train(40);
  int ok = 0;
  for (const auto& p : sm.phones()) {
    SDR r = sm.predict_sound(sm.articulatory(p));
    int best = -1;
    float best_ov = -1.0f;
    for (const auto& q : sm.phones()) {
      float ov = static_cast<float>(sm.acoustic(q).overlap(r));
      if (ov > best_ov) {
        best_ov = ov;
        best = static_cast<int>(&q - &sm.phones()[0]);
      }
    }
    if (best >= 0 && sm.phones()[static_cast<size_t>(best)] == p) ++ok;
  }
  CHECK(ok >= static_cast<int>(sm.phones().size()) - 2);

  Rng rng(77);
  SDR a = SDR::random(1024, 24, rng.next());
  SDR b = SDR::random(1024, 24, rng.next());
  SDR c = SDR::random(1024, 24, rng.next());
  SDR d = SDR::random(1024, 24, rng.next());
  auto shift = [](const SDR& s, size_t pos) {
    size_t sh = (pos % 4) * (s.size() / 4);
    SDR o(s.size(), 0);
    for (uint32_t i : s.indices()) o.set(static_cast<uint32_t>((i + sh) % s.size()));
    return o;
  };
  AssociativeMemory am(1024, 1024, 24, 0.1f);
  for (int e = 0; e < 80; ++e) {
    am.learn(shift(a, 0) | shift(b, 1), c);
    am.learn(shift(b, 0) | shift(a, 1), d);
  }
  SDR rab = am.recall(shift(a, 0) | shift(b, 1));
  SDR rba = am.recall(shift(b, 0) | shift(a, 1));
  CHECK(c.overlap(rab) > d.overlap(rab));
  CHECK(d.overlap(rba) > c.overlap(rba));
}

int main() {
  test_valuation();
  test_hippocampus();
  test_interoception();
  test_dual_controller();
  test_sp_prune();
  test_intero_value();
  test_feedback_api();
  test_var_order();
  test_words_assoc();
  test_speech_phase();
  return bmtest::summary("brain2");
}