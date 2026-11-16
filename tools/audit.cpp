#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/sdr.hpp"
#include "bm/speech.hpp"
#include "bm/interoception.hpp"
#include "bm/words.hpp"

using namespace bm;

namespace {

int Accuracy(const std::function<SDR(const std::string&)>& predict,
             const std::function<SDR(const std::string&)>& target,
             const std::vector<std::string>& items) {
  int ok = 0;
  for (const auto& it : items) {
    SDR r = predict(it);
    int best = -1;
    float best_ov = -1.0f;
    for (size_t j = 0; j < items.size(); ++j) {
      float ov = static_cast<float>(target(items[j]).overlap(r));
      if (ov > best_ov) {
        best_ov = ov;
        best = static_cast<int>(j);
      }
    }
    if (best >= 0 && items[static_cast<size_t>(best)] == it) ++ok;
  }
  return ok;
}

}  // namespace

int main() {
  // A. speech: is the loop real, or is a hand-written identity doing the work?
  {
    SpeechModel full(1024, 24, 31, true);
    SpeechModel noid(1024, 24, 31, false);
    full.train(200);
    noid.train(200);

    int full_ok = Accuracy([&](const std::string& p) { return full.predict_sound(full.articulatory(p)); },
                           [&](const std::string& p) { return full.acoustic(p); }, full.phones());
    int noid_ok = Accuracy([&](const std::string& p) { return noid.predict_sound(noid.articulatory(p)); },
                           [&](const std::string& p) { return noid.acoustic(p); }, noid.phones());
    printf("speech,with_identity=%d/%zu,features_only=%d/%zu\n", full_ok, full.phones().size(),
           noid_ok, noid.phones().size());

    // permuted labels: if the model can learn an arbitrary mapping, it is not hard-coded
    auto phones = full.phones();
    const size_t n = phones.size();
    AssociativeMemory am(1024, 1024, 24, 0.1f);
    for (int e = 0; e < 200; ++e) {
      for (size_t i = 0; i < n; ++i) {
        am.learn(full.articulatory(phones[i]), full.acoustic(phones[(i + 7) % n]));
      }
    }
    int perm_ok = 0;
    for (size_t i = 0; i < n; ++i) {
      SDR r = am.recall(full.articulatory(phones[i]));
      size_t target = (i + 7) % n;
      int best = -1;
      float best_ov = -1.0f;
      for (size_t j = 0; j < n; ++j) {
        float ov = static_cast<float>(full.acoustic(phones[j]).overlap(r));
        if (ov > best_ov) {
          best_ov = ov;
          best = static_cast<int>(j);
        }
      }
      if (best == static_cast<int>(target)) ++perm_ok;
    }
    printf("speech,permuted_labels=%d/%zu\n", perm_ok, n);
  }

  // B. language: does the word code contain the concept (leak), and can the model learn a
  // shuffled concept mapping?
  {
    WordEncoder we(512, 24, 11);
    double leak = 0;
    int nw = 0;
    std::vector<std::string> with_concepts;
    for (const auto& w : we.vocabulary()) {
      const WordEntry* e = we.entry(w);
      if (!e || e->concepts.empty()) continue;
      SDR cc(512, 0);
      for (const auto& c : e->concepts) cc |= we.concept_code(c);
      leak += static_cast<double>(we.encode(w).overlap(cc));
      ++nw;
      with_concepts.push_back(w);
    }
    printf("language,word_code_vs_concept_overlap=%.2f (code k=%zu)\n", nw ? leak / nw : 0.0,
           we.k());

    // shuffled mapping: learn word -> wrong concept; accuracy should follow the shuffle
    AssociativeMemory w2c(512, 512, 24, 0.08f);
    for (int e = 0; e < 8; ++e) {
      for (size_t i = 0; i < with_concepts.size(); ++i) {
        const WordEntry* ei = we.entry(with_concepts[i]);
        const WordEntry* ej = we.entry(with_concepts[(i + 3) % with_concepts.size()]);
        SDR cc(512, 0);
        for (const auto& c : ej->concepts) cc |= we.concept_code(c);
        (void)ei;
        w2c.learn(we.encode(with_concepts[i]), cc);
      }
    }
    int ok = 0;
    for (size_t i = 0; i < with_concepts.size(); ++i) {
      SDR r = w2c.recall(we.encode(with_concepts[i]));
      const WordEntry* ej = we.entry(with_concepts[(i + 3) % with_concepts.size()]);
      bool hit = false;
      for (const auto& c : ej->concepts) {
        if (we.concept_code(c).overlap(r) > 0) hit = true;
      }
      if (hit) ++ok;
    }
    printf("language,shuffled_word_to_concept=%d/%zu\n", ok, with_concepts.size());
  }

  // C. feeling: is body -> word learned, or is the word code leaking body state?
  {
    WordEncoder we(512, 24, 11);
    Interoception io(512);
    std::vector<std::pair<std::string, BodyState>> feelings = {
        {"bhookh", {92, 34, 15, 0.25f}},
        {"pyaas", {100, 30, 17, 0.35f}},
        {"thak", {68, 58, 12, 0.75f}},
        {"dard", {112, 26, 21, 0.85f}},
        {"khush", {76, 72, 14, 0.20f}},
    };
    double overlap = 0;
    for (const auto& f : feelings) overlap += static_cast<double>(io.encode(f.second).overlap(we.encode(f.first)));
    printf("feeling,body_word_overlap=%.2f\n", overlap / feelings.size());

    AssociativeMemory feel(512, 512, 24, 0.10f);
    for (int e = 0; e < 40; ++e) {
      for (size_t i = 0; i < feelings.size(); ++i) {
        feel.learn(io.encode(feelings[i].second),
                   we.encode(feelings[(i + 2) % feelings.size()].first));
      }
    }
    int ok = 0;
    for (size_t i = 0; i < feelings.size(); ++i) {
      SDR r = feel.recall(io.encode(feelings[i].second));
      auto ranked = [&] {
        std::vector<std::pair<std::string, float>> v;
        for (const auto& f : feelings) v.emplace_back(f.first, static_cast<float>(we.encode(f.first).overlap(r)));
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        return v;
      }();
      if (!ranked.empty() && ranked[0].first == feelings[(i + 2) % feelings.size()].first) ++ok;
    }
    printf("feeling,shuffled_body_to_word=%d/%zu\n", ok, feelings.size());

    // held-out grounding: train on all words except pyaas, test pyaas
    AssociativeMemory w2c(512, 512, 24, 0.08f);
    for (int e = 0; e < 8; ++e) {
      for (const auto& w : we.vocabulary()) {
        if (w == "pyaas") continue;
        const WordEntry* en = we.entry(w);
        if (!en || en->concepts.empty()) continue;
        SDR cc(512, 0);
        for (const auto& c : en->concepts) cc |= we.concept_code(c);
        w2c.learn(we.encode(w), cc);
      }
    }
    const WordEntry* py = we.entry("pyaas");
    SDR r = w2c.recall(we.encode("pyaas"));
    bool hit = false;
    for (const auto& c : py->concepts) {
      if (we.concept_code(c).overlap(r) > 0) hit = true;
    }
    printf("language,heldout_pyaas_concept_hit=%d\n", hit ? 1 : 0);
  }

  return 0;
}