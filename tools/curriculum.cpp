// Knowledge induction: facts, multi-hop inference, generalisation, forgetting.
// Everything is local Hebbian; no global loss, no backprop.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "bm/assoc.hpp"
#include "bm/encoder.hpp"
#include "bm/sdr.hpp"

using namespace bm;

namespace {

struct Hit {
  int rank = 1 << 20;
  float overlap = 0.0f;
  bool hit1() const { return rank == 0; }
  bool hit3() const { return rank < 3; }
};

Hit probe(const AssociativeMemory& m, const SDR& in, const SDR& target) {
  auto ranked = m.scores(in);
  std::vector<char> t(target.size(), 0);
  for (uint32_t i : target.indices()) t[i] = 1;
  Hit h;
  for (size_t r = 0; r < ranked.size(); ++r) {
    if (t[ranked[r].first]) {
      h.rank = static_cast<int>(r);
      break;
    }
  }
  SDR got = m.recall(in);
  h.overlap = target.count() ? float(target.overlap(got)) / float(target.count()) : 0.0f;
  return h;
}

struct Fact {
  std::string entity, relation, value;
};

std::vector<Fact> SetA() {
  return {
      {"gulab", "isa", "paudha"},   {"gulab", "has", "meetha"},
      {"gulab", "has", "kaanta"},   {"gulab", "lives", "baag"},
      {"machli", "isa", "jaanvar"}, {"machli", "lives", "paani"},
      {"machli", "has", "kaan"},    {"paani", "isa", "dravya"},
      {"paani", "has", "nirale"},
  };
}

std::vector<Fact> SetB() {
  return {
      {"kutta", "isa", "jaanvar"}, {"kutta", "lives", "ghar"},
      {"kutta", "has", "poonch"},  {"kutta", "has", "bhaunk"},
      {"billi", "isa", "jaanvar"}, {"billi", "lives", "ghar"},
      {"billi", "has", "poonch"},  {"roti", "isa", "khana"},
      {"roti", "has", "namak"},
  };
}

std::vector<Fact> SetC() {
  return {
      {"ped", "isa", "paudha"},  {"ped", "has", "taana"}, {"ped", "has", "chhaaya"},
      {"patthar", "isa", "dravya"}, {"patthar", "has", "sakht"},
      {"patthar", "has", "bhaari"},
  };
}

// cross-set inference facts, trained once and used as query keys
std::vector<Fact> SchemaFacts() {
  return {{"paudha", "needs", "paani"}, {"jaanvar", "needs", "khana"}};
}

std::vector<Fact> Interleave(const std::vector<Fact>& a, const std::vector<Fact>& b) {
  std::vector<Fact> out;
  size_t i = 0, j = 0;
  while (i < a.size() || j < b.size()) {
    if (i < a.size()) out.push_back(a[i++]);
    if (j < b.size()) out.push_back(b[j++]);
  }
  return out;
}

struct Brain {
  Encoder enc{512, 24, 7};
  std::map<std::string, AssociativeMemory> mem;
  bool by_features = false;

  AssociativeMemory& mem_for(const std::string& rel) {
    auto it = mem.find(rel);
    if (it == mem.end()) {
      it = mem.emplace(rel, AssociativeMemory(512, 512, 24, 0.05f)).first;
    }
    return it->second;
  }

  SDR entity(const std::string& name) {
    if (!by_features) return enc.symbol("ent:" + name);
    // feature-bundle entity code: shared features -> shared bits (generalisation)
    static const std::map<std::string, std::vector<std::string>> feat = {
        {"gulab", {"living", "plant", "soft", "sweet_smell"}},
        {"machli", {"living", "animal", "water", "swim"}},
        {"paani", {"liquid", "needed", "clear"}},
        {"kutta", {"living", "animal", "loyal", "bark"}},
        {"billi", {"living", "animal", "soft", "hunt"}},
        {"roti", {"food", "made", "needed"}},
        {"ped", {"living", "plant", "tall", "wood"}},
        {"patthar", {"hard", "heavy", "still"}},
    };
    std::vector<SDR> parts;
    auto it = feat.find(name);
    if (it != feat.end()) {
      for (const auto& f : it->second) parts.push_back(enc.symbol("feat:" + f));
    }
    parts.push_back(enc.symbol("ent:" + name));  // keep a private slice
    return enc.bundle(parts);
  }

  void train(const std::vector<Fact>& facts) {
    for (const auto& f : facts) {
      mem_for(f.relation).learn(entity(f.entity), enc.symbol("val:" + f.value));
    }
  }

  SDR value(const std::string& v) { return enc.symbol("val:" + v); }
};

struct Summary {
  int n = 0;
  int h1 = 0, h3 = 0;
  double mrr = 0.0, overlap = 0.0;
  void add(const Hit& h) {
    ++n;
    h1 += h.hit1();
    h3 += h.hit3();
    mrr += 1.0 / (1.0 + std::max(0, h.rank));
    overlap += h.overlap;
  }
  void report(const char* tag) const {
    printf("%-22s n=%2d hit1=%.3f hit3=%.3f mrr=%.3f overlap=%.3f\n", tag, n,
           n ? double(h1) / n : 0.0, n ? double(h3) / n : 0.0, n ? mrr / n : 0.0,
           n ? overlap / n : 0.0);
  }
};

Summary eval_set(Brain& b, const std::vector<Fact>& facts) {
  Summary s;
  for (const auto& f : facts) {
    auto& m = b.mem_for(f.relation);
    s.add(probe(m, b.entity(f.entity), b.value(f.value)));
  }
  return s;
}

// multi-hop: entity -> isa -> category, then category -> needs -> thing
Summary eval_multihop(Brain& b, const std::vector<std::pair<std::string, std::string>>& q) {
  Summary s;
  auto& isa = b.mem_for("isa");
  auto& need = b.mem_for("needs");
  for (const auto& [entity, expected] : q) {
    SDR cat = isa.recall(b.entity(entity));
    s.add(probe(need, cat, b.value(expected)));
  }
  return s;
}

// ---------------- experiment R: belief revision (non-monotonic update) ----------------
// Recall overlap of `in` against `target` at an explicit output width. k may exceed
// AssociativeMemory::k() when we want the superposition instead of only the winner.
double recall_overlap(const AssociativeMemory& m, const SDR& in, const SDR& target, size_t k) {
  SDR got = m.recall(in, k);
  return target.count() ? double(target.overlap(got)) / double(target.count()) : 0.0;
}

// "A isa B" is learned `reps` times, then "A isa C" is presented m times (m = 0..20).
// learn() only adds and applies a 1e-5 decay, so nothing is ever erased: the curve
// measures how readout competition moves, not how storage is overwritten.
void experiment_revision(bool features) {
  const int reps = 5;
  const int ms[] = {0, 1, 2, 5, 10, 20};
  const int last_m = ms[sizeof(ms) / sizeof(ms[0]) - 1];
  const size_t k = 24;
  const std::string ent = "gulab", old_v = "ped", new_v = "patthar";
  const std::string ctl = "machli", ctl_v = "paani";

  Brain rb;
  rb.by_features = features;
  for (int i = 0; i < reps; ++i) rb.train({{ent, "isa", old_v}});
  for (int i = 0; i < reps; ++i) rb.train({{ctl, "isa", ctl_v}});  // control fact, same matrix

  const SDR in = rb.entity(ent);
  const SDR old_code = rb.value(old_v);
  const SDR new_code = rb.value(new_v);
  const SDR ctl_in = rb.entity(ctl);
  const SDR ctl_code = rb.value(ctl_v);
  const double ctl_before = recall_overlap(rb.mem_for("isa"), ctl_in, ctl_code, k);

  printf("curr,revision_setup,entity=%s,old=%s,new=%s,reps=%d,k=%zu,control_before=%.3f\n",
         ent.c_str(), old_v.c_str(), new_v.c_str(), reps, k, ctl_before);

  int prev = 0;
  for (int m : ms) {
    for (int i = prev; i < m; ++i) rb.train({{ent, "isa", new_v}});
    prev = m;
    const AssociativeMemory& mem = rb.mem_for("isa");
    const double ro = recall_overlap(mem, in, old_code, k);
    const double rn = recall_overlap(mem, in, new_code, k);
    // recovery test on a copy: a single re-presentation of the old fact
    Brain cp = rb;
    cp.train({{ent, "isa", old_v}});
    const double ro1 = recall_overlap(cp.mem_for("isa"), in, old_code, k);
    printf("curr,revision,m=%d,recall_old=%.3f,recall_new=%.3f,recovered=%d\n", m, ro, rn,
           ro1 >= 0.5 ? 1 : 0);
    printf("curr,revision_re1,m=%d,recall_old_after1=%.3f\n", m, ro1);
    printf("curr,revision_wide,m=%d,k=%zu,recall_old=%.3f,recall_new=%.3f\n", m, 2 * k,
           recall_overlap(mem, in, old_code, 2 * k), recall_overlap(mem, in, new_code, 2 * k));
    printf("curr,revision_control,m=%d,entity=%s,recall=%.3f\n", m, ctl.c_str(),
           recall_overlap(mem, ctl_in, ctl_code, k));
  }

  // how many re-presentations of the old fact are needed to win the readout back?
  Brain rp = rb;
  int need = 0;
  for (int r = 1; r <= 64; ++r) {
    rp.train({{ent, "isa", old_v}});
    if (recall_overlap(rp.mem_for("isa"), in, old_code, k) >= 0.9) {
      need = r;
      break;
    }
  }
  printf("curr,revision_recover,reps_to_overlap0.9=%d,verdict=%s\n", need,
         need > 0 ? "recoverable" : "not_recoverable");

  // control: revision is local, the other entity's fact must survive it
  const double ctl_after = recall_overlap(rb.mem_for("isa"), ctl_in, ctl_code, k);
  printf("curr,revision_control_summary,entity=%s,recall_before=%.3f,recall_after=%.3f\n",
         ctl.c_str(), ctl_before, ctl_after);
  // permuted-output control: untrained codes must sit at chance, averaged over 8 draws
  double perm_sum = 0.0;
  const int nperm = 8;
  for (int i = 0; i < nperm; ++i) {
    perm_sum += recall_overlap(rb.mem_for("isa"), in, rb.value("__permute" + std::to_string(i)), k);
  }
  printf("curr,revision_permuted,m=%d,recall_unrelated=%.3f,chance=%.3f,n=%d\n", last_m,
         perm_sum / nperm, double(k) / 512.0, nperm);
}

// ---------------- experiment H: reasoning depth ----------------
struct DepthStat {
  int n = 0, top1 = 0;
  double overlap = 0.0, rank = 0.0;
  void add(const Hit& h) {
    ++n;
    top1 += h.hit1() ? 1 : 0;
    overlap += h.overlap;
    rank += h.rank;
  }
};

// Chains of entity codes A -> B -> C -> D -> E in one relation ("hop"). Each hop feeds
// the previous hop's recall straight back in as the next input: iterated associative
// recall, no labels, no search. `shared` puts every chain in one matrix (a knowledge
// base where entities have several successors) instead of one matrix per chain.
std::vector<DepthStat> run_hops(bool features, const std::vector<std::vector<std::string>>& chains,
                                bool shuffled, int skip_link, bool shared) {
  const std::vector<std::string> pool = {"gulab", "machli", "paani", "kutta",
                                         "billi", "roti",   "ped",    "patthar"};
  std::vector<DepthStat> st(6);
  Rng rng(0x5EED1234ull);
  Brain shared_brain;
  shared_brain.by_features = features;
  for (const auto& chain : chains) {
    Brain solo;
    solo.by_features = features;
    Brain& hb = shared ? shared_brain : solo;
    AssociativeMemory& mem = hb.mem_for("hop");
    for (size_t i = 0; i + 1 < chain.size(); ++i) {
      if (skip_link >= 0 && static_cast<int>(i) == skip_link) continue;
      SDR in = shuffled ? hb.entity(pool[rng.below(uint32_t(pool.size()))]) : hb.entity(chain[i]);
      SDR out = shuffled ? hb.enc.symbol("noise:" + std::to_string(rng.below(1000000)))
                         : hb.entity(chain[i + 1]);
      mem.learn(in, out);
    }
    SDR cur = hb.entity(chain[0]);
    for (size_t d = 1; d < chain.size() && d < st.size(); ++d) {
      st[d].add(probe(mem, cur, hb.entity(chain[d])));
      cur = mem.recall(cur);
    }
  }
  return st;
}

void report_hops(const char* tag, const std::vector<DepthStat>& st) {
  for (size_t d = 1; d < st.size(); ++d) {
    if (!st[d].n) continue;
    printf("curr,%s,depth=%zu,top1=%.3f,overlap=%.3f,rank=%.2f\n", tag, d,
           double(st[d].top1) / st[d].n, st[d].overlap / st[d].n, st[d].rank / st[d].n);
  }
}

void experiment_hops(bool features) {
  const std::vector<std::string> pool = {"gulab", "machli", "paani", "kutta",
                                         "billi", "roti",   "ped",    "patthar"};
  // deterministic per-chain shuffles, so an entity has a *different* successor in
  // different chains (a knowledge base, not one long path)
  std::vector<std::vector<std::string>> chains;
  for (size_t c = 0; c < 24; ++c) {
    std::vector<std::string> ch = pool;
    Rng rng(0xABCD0000ull + c * 7919ull);
    for (size_t i = ch.size(); i > 1; --i) std::swap(ch[i - 1], ch[rng.below(uint32_t(i))]);
    ch.resize(5);
    chains.push_back(ch);
  }
  const double chance = 24.0 / 512.0;  // k / n_out: expected overlap of two random codes
  // Why one code per node: "val:X" and "ent:X" are different codes, so a hop that outputs
  // a value and feeds it to a memory keyed by entities only works through a shared bit.
  {
    Encoder enc(512, 24, 7);
    const int npair = 64;
    int zero = 0;
    double shared = 0.0;
    for (int i = 0; i < npair; ++i) {
      const std::string s = "sym" + std::to_string(i);
      const size_t o = enc.symbol("val:" + s).overlap(enc.symbol("ent:" + s));
      shared += double(o);
      if (!o) ++zero;
    }
    printf("curr,hops_namespace,pairs=%d,mean_shared_bits=%.2f,zero_overlap_frac=%.3f\n", npair,
           shared / npair, double(zero) / npair);
  }
  printf("curr,hops_setup,chains=%zu,links_per_chain=%zu,isolated_matrix=1\n", chains.size(),
         chains[0].size() - 1);
  printf("curr,hops_chance,top1=%.3f,overlap=%.3f\n", chance, chance);
  report_hops("hops", run_hops(features, chains, false, -1, false));
  report_hops("hops_shared", run_hops(features, chains, false, -1, true));
  report_hops("hops_ablation", run_hops(features, chains, false, 2, false));
  report_hops("hops_ablation_tail", run_hops(features, chains, false, 3, false));
  report_hops("hops_control", run_hops(features, chains, true, -1, false));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "noreplay";
  const bool replay = mode == "replay";
  const bool inter = mode == "interleave";
  bool features = argc > 2 && std::string(argv[2]) == "features";

  auto A = SetA(), B = SetB(), C = SetC(), S = SchemaFacts();
  std::vector<std::string> a_ents = {"gulab", "machli", "paani"};

  printf("mode: entity=%s replay=%s\n", features ? "feature-bundle" : "hash", mode.c_str());

  Brain b;
  b.by_features = features;
  b.train(S);

  Summary direct;
  b.train(A);
  direct = eval_set(b, A);
  direct.report("A direct (fresh)");
  Summary mh_fresh = eval_multihop(b, {{"gulab", "paani"}, {"machli", "khana"}});
  mh_fresh.report("A multihop (fresh)");

  if (inter) {
    b.train(Interleave(A, B));
  } else {
    b.train(B);
  }
  Summary afterB = eval_set(b, A);
  afterB.report("A after B");
  Summary mh_B = eval_multihop(b, {{"gulab", "paani"}, {"machli", "khana"}});
  mh_B.report("A multihop after B");

  if (replay) {
    for (int i = 0; i < 6; ++i) b.train(A);
  }

  if (inter) {
    b.train(Interleave(A, C));
  } else {
    b.train(C);
  }
  Summary afterC = eval_set(b, A);
  afterC.report("A after C");
  Summary mh_C = eval_multihop(b, {{"gulab", "paani"}, {"machli", "khana"}});
  mh_C.report("A multihop after C");
  Summary b_ret = eval_set(b, B);
  b_ret.report("B after C");
  Summary c_direct = eval_set(b, C);
  c_direct.report("C direct");
  Summary mh_Cc = eval_multihop(b, {{"ped", "paani"}, {"patthar", "khana"}});
  mh_Cc.report("C multihop");

  // shuffled control: same facts, random outputs
  if (!features) {
    Brain sh;
    sh.by_features = false;
    for (const auto& f : A) {
      sh.mem_for(f.relation).learn(sh.entity(f.entity),
                                   sh.enc.symbol("noise:" + std::to_string(rand() % 100000)));
    }
    Summary shs = eval_set(sh, A);
    shs.report("A shuffled control");
  }

  experiment_revision(features);
  experiment_hops(features);

  return 0;
}