#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/sdr.hpp"
#include "bm/speech.hpp"

using namespace bm;

namespace {

SDR PhaseShift(const SDR& s, size_t pos, size_t gamma) {
  size_t shift = (pos % gamma) * (s.size() / gamma);
  SDR o(s.size(), 0);
  for (uint32_t i : s.indices()) o.set(static_cast<uint32_t>((i + shift) % s.size()));
  return o;
}

SDR Noise(const SDR& s, int flips, Rng& rng) {
  SDR o = s;
  for (int i = 0; i < flips; ++i) {
    auto idx = s.indices();
    if (idx.empty()) break;
    o.reset(idx[rng.below(static_cast<uint32_t>(idx.size()))]);
  }
  for (int i = 0; i < flips; ++i) {
    for (int guard = 0; guard < 64; ++guard) {
      uint32_t bit = rng.below(static_cast<uint32_t>(s.size()));
      if (!o.test(bit)) {
        o.set(bit);
        break;
      }
    }
  }
  return o;
}

std::vector<std::pair<std::string, float>> Rank(const SDR& r, const SpeechModel& sm) {
  std::vector<std::pair<std::string, float>> v;
  for (const auto& p : sm.phones()) {
    v.emplace_back(p, static_cast<float>(sm.acoustic(p).overlap(r)));
  }
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  return v;
}

std::vector<std::pair<std::string, float>> RankArt(const SDR& r, const SpeechModel& sm) {
  std::vector<std::pair<std::string, float>> v;
  for (const auto& p : sm.phones()) {
    v.emplace_back(p, static_cast<float>(sm.articulatory(p).overlap(r)));
  }
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  return v;
}

}  // namespace

int main() {
  SpeechModel sm(1024, 24, 31);
  sm.train(200);

  // 1. sensorimotor loop: articulation -> sound and sound -> articulation
  int fwd_ok = 0, inv_ok = 0, n = 0;
  for (const auto& p : sm.phones()) {
    ++n;
    auto rs = Rank(sm.predict_sound(sm.articulatory(p)), sm);
    auto ra = RankArt(sm.predict_art(sm.acoustic(p)), sm);
    if (!rs.empty() && rs[0].first == p) ++fwd_ok;
    if (!ra.empty() && ra[0].first == p) ++inv_ok;
  }
  printf("sensorimotor,phones=%d,forward_top1=%.3f,inverse_top1=%.3f\n", n,
         static_cast<double>(fwd_ok) / n, static_cast<double>(inv_ok) / n);

  // 2. robustness to articulation noise
  {
    Rng rng(41);
    for (int flips : {2, 4, 8}) {
      int ok = 0;
      for (const auto& p : sm.phones()) {
        SDR art = Noise(sm.articulatory(p), flips, rng);
        auto r = Rank(sm.predict_sound(art), sm);
        if (!r.empty() && r[0].first == p) ++ok;
      }
      printf("noise,flips=%d,forward_top1=%.3f\n", flips, static_cast<double>(ok) / n);
    }
  }

  // 3. confusability: similar articulation and similar sound
  {
    printf("confusion,art_p_b=%.1f,art_t_d=%.1f,snd_ee_ii=%.1f\n",
           static_cast<double>(sm.articulatory("p").overlap(sm.articulatory("b"))),
           static_cast<double>(sm.articulatory("t").overlap(sm.articulatory("d"))),
           static_cast<double>(sm.acoustic("ee").overlap(sm.acoustic("ii"))));
  }

  // 4. phase coding: same items, different order must be separable
  {
    Rng rng(77);
    std::vector<SDR> item(4);
    const char* names[4] = {"A", "B", "C", "D"};
    for (int i = 0; i < 4; ++i) item[i] = SDR::random(1024, 24, rng.next());

    AssociativeMemory no_phase(1024, 1024, 24, 0.1f);
    AssociativeMemory phase(1024, 1024, 24, 0.1f);
    for (int e = 0; e < 80; ++e) {
      no_phase.learn(item[0] | item[1], item[2]);  // AB -> C
      no_phase.learn(item[1] | item[0], item[3]);  // BA -> D
      phase.learn(PhaseShift(item[0], 0, 4) | PhaseShift(item[1], 1, 4), item[2]);
      phase.learn(PhaseShift(item[1], 0, 4) | PhaseShift(item[0], 1, 4), item[3]);
    }
    auto top = [](const SDR& r, const std::vector<SDR>& it) {
      int best = -1;
      float best_ov = -1;
      for (size_t i = 0; i < it.size(); ++i) {
        float ov = static_cast<float>(it[i].overlap(r));
        if (ov > best_ov) {
          best_ov = ov;
          best = static_cast<int>(i);
        }
      }
      return best;
    };
    SDR c_ab = no_phase.recall(item[0] | item[1]);
    SDR c_ba = no_phase.recall(item[1] | item[0]);
    SDR p_ab = phase.recall(PhaseShift(item[0], 0, 4) | PhaseShift(item[1], 1, 4));
    SDR p_ba = phase.recall(PhaseShift(item[1], 0, 4) | PhaseShift(item[0], 1, 4));
    printf("order,no_phase_AB=%s,no_phase_BA=%s,phase_AB=%s,phase_BA=%s\n", names[top(c_ab, item)],
           names[top(c_ba, item)], names[top(p_ab, item)], names[top(p_ba, item)]);
    printf("order_context_overlap,no_phase=%.1f,phase=%.1f\n",
           static_cast<double>((item[0] | item[1]).overlap(item[1] | item[0])),
           static_cast<double>((PhaseShift(item[0], 0, 4) | PhaseShift(item[1], 1, 4))
                                   .overlap(PhaseShift(item[1], 0, 4) | PhaseShift(item[0], 1, 4))));
  }

  return 0;
}