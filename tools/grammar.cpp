// Minimal-pair grammar benchmark (forced choice), BLiMP-style.
//
//   grammar <corpus> <V> <max_tokens> <offsets> <xi> <use_cache> <beta> <lam> <ctx_mode>
//           <trace_k> <gamma> <code_mode> <feat_repeat> <freq_floor> <feat_boost>
//           <readout_mode> <readout_temp> <pos_w> <head_mode> <head_w>
//
// The model is a next-word predictor, so "does it know grammar" is measured the only way that is
// fair to it: given the context, does it score the grammatical continuation above the
// ungrammatical one? Chance is 50%. The pairs are hand-written minimal pairs, because no POS-tagged
// corpus ships with this repo and a template over an untagged corpus would silently test nothing.
//
// Two context modes are compared:
//   offsets = consecutive positions, e.g. "1,2,3"
//   offsets = skip positions, e.g. "1,2,3,4,6,8" - the subject can then sit at distance 4-8 while
//             the verb is at distance 1, which is what agreement actually needs.
//
// Reported: per-phenomenon accuracy, overall accuracy, and coverage (how many pairs have both
// candidates inside the vocabulary). A low coverage number means the test is not measuring grammar.

#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "bm/ngram_cache.hpp"
#include "bm/sdr.hpp"

namespace {

struct Pair {
  std::string phenomenon;
  std::vector<std::string> ctx;
  std::string good;
  std::string bad;
  bool amb;  // context carries two number-bearing words that disagree, so the correct answer
             // cannot be recovered from a bag of features alone
};

// Minimal pairs. Every context is grammatical English and the two continuations differ only in the
// feature under test, so a model that only tracks local bigrams scores chance.
std::vector<Pair> MakePairs() {
  std::vector<Pair> p;
  auto add = [&](const char* ph, std::vector<std::string> ctx, std::string g, std::string b,
                 bool amb = false) {
    p.push_back({ph, std::move(ctx), std::move(g), std::move(b), amb});
  };

  // subject-verb agreement, subject and verb adjacent
  add("agreement_adjacent", {"the", "keys", "on", "the", "table"}, "turn", "turns");
  add("agreement_adjacent", {"the", "key", "on", "the", "table"}, "turns", "turn");
  add("agreement_adjacent", {"the", "boxes", "in", "the", "room"}, "are", "is");
  add("agreement_adjacent", {"the", "box", "in", "the", "room"}, "is", "are");
  add("agreement_adjacent", {"the", "children", "in", "the", "park"}, "were", "was");
  add("agreement_adjacent", {"the", "child", "in", "the", "park"}, "was", "were");
  add("agreement_adjacent", {"the", "teams", "with", "the", "new", "coach"}, "have", "has");
  add("agreement_adjacent", {"the", "team", "with", "the", "new", "coach"}, "has", "have");

  // agreement at a distance: a relative clause sits between subject and verb
  add("agreement_long", {"the", "keys", "that", "the", "man", "found"}, "turn", "turns", true);
  add("agreement_long", {"the", "key", "that", "the", "man", "found"}, "turns", "turn", true);
  add("agreement_long", {"the", "boxes", "that", "the", "children", "opened"}, "were", "was");
  add("agreement_long", {"the", "box", "that", "the", "children", "opened"}, "was", "were");
  // Past "left" cannot test number: it is invariant, and it is also grammatical after a singular
  // subject, so "the man who the women saw left" has two right answers. Use the present pair
  // leave/leaves, which contrasts number and is the form the corpus actually teaches.
  add("agreement_long", {"the", "men", "who", "the", "women", "saw"}, "leave", "leaves");
  add("agreement_long", {"the", "man", "who", "the", "women", "saw"}, "leaves", "leave");

  // determiner-noun number
  add("determiner_noun", {"this"}, "car", "cars", true);
  add("determiner_noun", {"these"}, "cars", "car", true);
  add("determiner_noun", {"that"}, "house", "houses", true);
  add("determiner_noun", {"those"}, "houses", "house", true);
  add("determiner_noun", {"many"}, "waters", "water");
  add("determiner_noun", {"much"}, "water", "waters");

  // auxiliary agreement
  add("aux_agreement", {"he"}, "is", "are");
  add("aux_agreement", {"they"}, "are", "is");
  add("aux_agreement", {"the", "team"}, "has", "have");
  add("aux_agreement", {"the", "teams"}, "have", "has");

  // "one of the X" takes a singular verb
  add("one_of", {"one", "of", "the", "keys"}, "turns", "turn", true);
  add("one_of", {"one", "of", "the", "boxes"}, "is", "are", true);

  // polarity: a negative auxiliary blocks "some" but allows "any"
  add("polarity", {"we", "did", "not", "see"}, "any", "some");
  add("polarity", {"we", "saw"}, "some", "any");

  return p;
}

// Held-out pairs. A pair file lets the test run on combinations the model was tuned on and on
// combinations it never saw, which is the only way to tell a learned rule from a memorised table.
// Format per line: phenomenon|context words|good|bad|amb
std::vector<Pair> LoadPairs(const std::string& path) {
  std::vector<Pair> out;
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open pair file %s\n", path.c_str());
    std::exit(2);
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> f;
    size_t start = 0;
    while (start <= line.size()) {
      size_t bar = line.find('|', start);
      if (bar == std::string::npos) bar = line.size();
      f.push_back(line.substr(start, bar - start));
      if (bar == line.size()) break;
      start = bar + 1;
    }
    if (f.size() < 5) continue;
    Pair pr;
    pr.phenomenon = f[0];
    pr.good = f[2];
    pr.bad = f[3];
    pr.amb = f[4] == "1";
    std::istringstream cs(f[1]);
    std::string w;
    while (cs >> w) pr.ctx.push_back(w);
    out.push_back(std::move(pr));
  }
  return out;
}

// Hand-labelled features for the closed-class words that appear in the minimal pairs.
// A random superposition code cannot express "this noun is plural" as a thing the matrix can read
// out: plurality is smeared over all 24 bits and shares them with meaning. These bit-ranges give the
// matrix one bit per feature, so a single second-order term can carry the whole rule.
enum Feat : uint32_t {
  F_PLURAL = 0,
  F_SINGULAR,
  F_VERB3SG,
  F_VERBBASE,
  F_PAST,
  F_DET_SG,
  F_DET_PL,
  F_NEG,
  F_POS,
  F_COUNT
};

const std::unordered_map<std::string, std::vector<Feat>>& FeatureTable() {
  static const std::unordered_map<std::string, std::vector<Feat>> t = {
      {"keys", {F_PLURAL}},      {"boxes", {F_PLURAL}},    {"children", {F_PLURAL}},
      {"teams", {F_PLURAL}},     {"men", {F_PLURAL}},      {"files", {F_PLURAL}},
      {"cars", {F_PLURAL}},      {"houses", {F_PLURAL}},   {"waters", {F_PLURAL}},
      // The table must cover the whole lexicon, not just the words the tuned pairs happen to use.
      // A noun that is missing here gets its number from the -s rule only, and "women", "coach",
      // "room", "table", "park" have no -s at all, so they carried no number feature.
      {"coaches", {F_PLURAL}},   {"rooms", {F_PLURAL}},    {"tables", {F_PLURAL}},
      {"parks", {F_PLURAL}},     {"women", {F_PLURAL}},
      {"key", {F_SINGULAR}},     {"box", {F_SINGULAR}},    {"child", {F_SINGULAR}},
      {"team", {F_SINGULAR}},    {"man", {F_SINGULAR}},    {"file", {F_SINGULAR}},
      {"car", {F_SINGULAR}},     {"house", {F_SINGULAR}},  {"water", {F_SINGULAR}},
      {"coach", {F_SINGULAR}},   {"room", {F_SINGULAR}},   {"table", {F_SINGULAR}},
      {"park", {F_SINGULAR}},    {"woman", {F_SINGULAR}},
      {"turn", {F_VERBBASE}},    {"are", {F_VERBBASE}},    {"were", {F_VERBBASE}},
      {"have", {F_VERBBASE}},    {"left", {F_VERBBASE}},   {"work", {F_VERBBASE}},
      {"leave", {F_VERBBASE}},
      {"turns", {F_VERB3SG}},    {"is", {F_VERB3SG}},      {"was", {F_VERB3SG, F_PAST}},
      {"has", {F_VERB3SG}},      {"leaves", {F_VERB3SG}},
      // Every 3sg form must be listed. A 3sg verb that is not in the table falls through to the
      // -s morphology rule and is labelled plural, which makes a plural head predict it. That is
      // the whole held-out failure: works/opens/finds were plural, so "the files ... works".
      {"works", {F_VERB3SG}},    {"opens", {F_VERB3SG}},   {"finds", {F_VERB3SG}},
      {"open", {F_VERBBASE}},    {"find", {F_VERBBASE}},
      {"this", {F_DET_SG}},      {"that", {F_DET_SG}},     {"much", {F_DET_SG}},
      {"these", {F_DET_PL}},     {"those", {F_DET_PL}},    {"many", {F_DET_PL}},
      {"some", {F_POS}},         {"any", {F_NEG}},
      {"not", {F_NEG}},
      {"he", {F_SINGULAR}},   {"she", {F_SINGULAR}},  {"it", {F_SINGULAR}},
      {"i", {F_SINGULAR}},
      {"they", {F_PLURAL}},   {"we", {F_PLURAL}},     {"you", {F_PLURAL}},
  };
  return t;
}

// Morphological fallback: a word that ends in -s but is not a closed-class form is plural. This
// is what lets a novel noun get a feature at test time instead of having to be listed by hand.
const std::unordered_set<std::string>& OpenClass() {
  static const std::unordered_set<std::string> t = {
      "is", "was", "has", "does", "this", "these", "those", "yes", "his", "its", "was", "thus",
      "always", "perhaps", "nevertheless", "news", "series", "species"};
  return t;
}

std::vector<Feat> FeaturesOf(const std::string& w, int code_mode) {
  std::vector<Feat> out;
  const auto& tab = FeatureTable();
  auto it = tab.find(w);
  if (it != tab.end()) {
    out = it->second;
    if (code_mode == 3 && it->second.empty()) {
      // the table entry may be empty for an open-class word; fall through to morphology
    } else {
      return out;
    }
  }
  if (w.size() > 2 && w.back() == 's' && !OpenClass().count(w)) out.push_back(F_PLURAL);
  if (w.size() > 3 && w.compare(w.size() - 3, 3, "ing") == 0) out.push_back(F_VERBBASE);
  return out;
}

std::vector<std::string> Tokenize(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'') {
      cur.push_back(c);
    } else {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "data/wikitext103_train.txt";
  const size_t V = argc > 2 ? std::stoul(argv[2]) : 10000;
  const long max_tokens = argc > 3 ? std::stol(argv[3]) : 2000000;
  const std::string offsets_arg = argc > 4 ? argv[4] : "1,2,3";
  const float xi = argc > 5 ? std::stof(argv[5]) : 0.005f;
  const int use_cache = argc > 6 ? std::stoi(argv[6]) : 1;
  const float cache_beta = argc > 7 ? std::stof(argv[7]) : 0.5f;
  const float lam = argc > 8 ? std::stof(argv[8]) : 10.0f;
  // context mode: 0 = fixed word slots, 1 = decayed trace (stateful, unbounded time depth)
  const int ctx_mode = argc > 9 ? std::stoi(argv[9]) : 0;
  const int trace_k = argc > 10 ? std::stoi(argv[10]) : 48;
  const float gamma = argc > 11 ? std::stof(argv[11]) : 0.9f;

  std::vector<int> offsets;
  {
    size_t start = 0;
    while (start < offsets_arg.size()) {
      size_t comma = offsets_arg.find(',', start);
      if (comma == std::string::npos) comma = offsets_arg.size();
      offsets.push_back(std::stoi(offsets_arg.substr(start, comma - start)));
      start = comma + 1;
    }
    std::sort(offsets.begin(), offsets.end());
  }
  const int max_off = offsets.back();
  const size_t N = 512, K = 24;

  const auto t0 = std::chrono::steady_clock::now();

  std::ifstream in(path);
  if (!in) {
    printf("cannot open %s\n", path.c_str());
    return 1;
  }
  std::unordered_map<std::string, long> freq;
  freq.reserve(400000);
  std::string line;
  long n_tokens = 0, n_sents = 0;
  while (std::getline(in, line)) {
    auto toks = Tokenize(line);
    if (toks.size() < 3) continue;
    ++n_sents;
    for (const auto& t : toks) ++freq[t];
    ++n_tokens;
  }
  printf("corpus=%s sentences=%ld tokens=%ld types=%zu\n", path.c_str(), n_sents, n_tokens,
         freq.size());

  std::vector<std::pair<std::string, long>> by_freq(freq.begin(), freq.end());
  std::sort(by_freq.begin(), by_freq.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::unordered_map<std::string, int> index;
  std::vector<std::string> vocab_label;
  vocab_label.reserve(std::min(V, by_freq.size()) + 1);
  index.reserve(std::min(V, by_freq.size()) * 2 + 8);
  for (size_t i = 0; i < std::min(V, by_freq.size()); ++i) {
    index[by_freq[i].first] = static_cast<int>(i);
    vocab_label.push_back(by_freq[i].first);
  }
  vocab_label.push_back("<unk>");
  const int UNK = static_cast<int>(std::min(V, by_freq.size()));
  auto code_of = [&](const std::string& w) {
    auto it = index.find(w);
    return it == index.end() ? UNK : it->second;
  };

  const int code_mode = argc > 12 ? std::stoi(argv[12]) : 0;
  const int feat_repeat = argc > 13 ? std::stoi(argv[13]) : 16;
  const float freq_floor = argc > 14 ? std::stof(argv[14]) : 0.05f;
  // Feature bits are the structure the model is told about, identity bits are episodic detail.
  // They do not deserve the same gain: at equal gain the identity bias (~2.4) is as large as the
  // agreement differential (~3) and they cancel.
  const float feat_boost = argc > 15 ? std::stof(argv[15]) : 8.0f;
  const float pos_w = argc > 18 ? std::stof(argv[18]) : 1.0f;
  const int readout_mode = argc > 16 ? std::stoi(argv[16]) : 0;
  const float readout_temp = argc > 17 ? std::stof(argv[17]) : 1.0f;
  const size_t kIdent = 12;      // identity bits
  const size_t kFeatBase = 256;  // feature bits start here, out of N=512
  std::vector<std::vector<uint32_t>> idxb;
  idxb.reserve(UNK + 1);

  std::vector<bm::SDR> codes;
  codes.reserve(UNK + 1);
  for (int v = 0; v <= UNK; ++v) {
    // SDR(n,k) seeds its RNG from n and k only, so every word would get the same code.
    // Seed per word, or the readout cannot tell two words apart.
    codes.push_back(bm::SDR::random(N, K, 0x9e3779b97f4a7c15ull * static_cast<uint64_t>(v + 1)));
    idxb.emplace_back(codes.back().indices());
    if (code_mode >= 4 && v < UNK) idxb.back().clear();
    if (code_mode >= 2 && v < UNK) {
      // hybrid: keep the random identity bits in the low half, then add one bit per feature so the
      // matrix can read a feature out directly
      const std::vector<Feat> fs = FeaturesOf(vocab_label[static_cast<size_t>(v)], code_mode);
      // Feature-only codes were too aggressive: a word with no feature had NO bits at all, so it
      // was invisible to the readout and could not serve as a head. That is how "one" was lost.
      // Giving every word a small identity component fixes that without bringing back the 12-bit
      // per-word bias that D53 measured: 4 bits of identity against 16 feature bits is a 20%
      // identity share instead of a third.
      const size_t id_bits = (code_mode < 4) ? kIdent : 4;
      bm::SDR ident =
          bm::SDR::random(kFeatBase, id_bits, 0x9e3779b97f4a7c15ull * static_cast<uint64_t>(v + 7));
      for (uint32_t i : ident.indices()) idxb[static_cast<size_t>(v)].push_back(i);
      for (Feat f : fs) {
        for (int r = 0; r < feat_repeat; ++r) {
          const size_t b = kFeatBase + static_cast<size_t>(f) * feat_repeat + r;
          if (b < N) idxb[static_cast<size_t>(v)].push_back(static_cast<uint32_t>(b));
        }
      }
    }
  }
  if (code_mode == 2) {
    int applied = 0;
    for (const auto& kv : FeatureTable()) {
      auto it = index.find(kv.first);
      if (it == index.end()) continue;
      const int v = it->second;
      idxb[static_cast<size_t>(v)].clear();
      // identity bits: 18 of the first 384, but drop the ones colliding with features
      if (code_mode < 4) {
        bm::SDR ident = bm::SDR::random(kFeatBase, kIdent,
                                        0x9e3779b97f4a7c15ull * static_cast<uint64_t>(v + 7));
        for (uint32_t i : ident.indices()) idxb[static_cast<size_t>(v)].push_back(i);
      } else if (FeaturesOf(vocab_label[static_cast<size_t>(v)], code_mode).empty()) {
        // a word with no feature would otherwise have an empty code and vanish from the context;
        // give it a small identity code so it still occupies a position
        bm::SDR ident = bm::SDR::random(kFeatBase, 4,
                                        0x9e3779b97f4a7c15ull * static_cast<uint64_t>(v + 7));
        for (uint32_t i : ident.indices()) idxb[static_cast<size_t>(v)].push_back(i);
      }
      // a feature is repeated: one bit out of twelve cannot outvote the identity bits, and the
      // whole point is that the matrix should read the feature, not the word
      for (Feat f : FeaturesOf(kv.first, code_mode)) {
        for (int r = 0; r < feat_repeat; ++r) {
          const size_t b = kFeatBase + static_cast<size_t>(f) * feat_repeat + r;
          if (b < N) idxb[static_cast<size_t>(v)].push_back(static_cast<uint32_t>(b));
        }
      }
      ++applied;
    }
    printf("hybrid_code_words=%d of %zu\n", applied, FeatureTable().size());
  }
  auto bits_of = [&](int v) -> const std::vector<uint32_t>& {
    return idxb[static_cast<size_t>(v < 0 ? UNK : v)];
  };

  // One weight matrix per context slot, plus one extra "bag" matrix that sees the union of all
  // context bits. The slots carry position and the bag carries pooled co-occurrence: on a small
  // lexicon the slots alone are enough, but on real text splitting the data six ways starves each
  // slot, so the bag is what keeps the general statistics alive.
  const size_t n_slot = offsets.size();
  const size_t bag = n_slot;
  const size_t n_mat = n_slot + 1;
  std::vector<float> w(n_mat * N * N, 0.0f);
  std::vector<float> omega(n_mat * N * N, 0.0f);
  // Head stack: on a relativiser the current noun becomes the head of the clause, and the verb that
  // closes the clause reads that head. This is the translation-invariant association the distance
  // slots cannot express: the head is bound by identity, not by how far away it sits.
  const int head_mode = argc > 19 ? std::stoi(argv[19]) : 0;
  const float head_w = argc > 20 ? std::stof(argv[20]) : 1.0f;
  // How much the distance slots still count once a head is known. 1.0 keeps the old behaviour
  // (slots vote as usual); 0.0 lets the head decide alone.
  const float slot_head_w = argc > 21 ? std::stof(argv[21]) : 0.25f;
  const std::string pair_file = argc > 22 ? argv[22] : "";
  const std::vector<Pair> pairs = pair_file.empty() ? MakePairs() : LoadPairs(pair_file);
  std::vector<float> w_head(head_mode ? N * N : 0, 0.0f);
  std::vector<float> omega_head(head_mode ? N * N : 0, 0.0f);
  std::vector<int> head_stack;
  int last_noun = UNK;
  long g_head_updates = 0;
  long g_head_one = 0;
  long g_push_one = 0;

  auto is_relativiser = [](const std::string& w) {
    return w == "that" || w == "who" || w == "which";
  };
  // A preposition opens a modifier phrase: the noun that follows it is inside that phrase and is
  // not the subject. Without this, "the keys on the table" makes "table" the head and the model
  // agrees with the nearest noun, which is exactly the attraction error.
  auto is_prep = [](const std::string& w) {
    return w == "in" || w == "on" || w == "with" || w == "near" || w == "of" || w == "for" ||
           w == "by" || w == "at" || w == "from";
  };
  bool in_pp = false;
  auto is_noun = [&](int v) {
    if (v == UNK) return false;
    const auto& fs = FeaturesOf(vocab_label[static_cast<size_t>(v)], code_mode);
    for (Feat f : fs) {
      if (f == F_PLURAL || f == F_SINGULAR) return true;
    }
    return false;
  };
  auto push_head = [&](int v) {
    if (!head_mode || v == UNK) return;
    const std::string& lab = vocab_label[static_cast<size_t>(v)];
    if (is_relativiser(lab)) {
      if (last_noun != UNK && head_stack.size() < 8) head_stack.push_back(last_noun);
    } else if (lab == "one") {
      ++g_push_one;
      // "one of the keys" takes a singular verb: the head of that clause is the quantifier, not the
      // plural noun that follows it. Pushing "one" is the same trick as the relativiser case, only
      // the trigger is a quantifier instead of a complementiser.
      if (head_stack.size() < 8) head_stack.push_back(v);
    } else if (is_noun(v)) {
      if (!in_pp) last_noun = v;
    } else if (is_prep(lab)) {
      in_pp = true;
    }
  };
  auto head_bits = [&]() -> int {
    // A relativiser or a quantifier pushes an explicit head. Otherwise the head is simply the last
    // noun that is not inside a modifier phrase, which is the subject. Without this fallback the
    // model has no head at all in "the keys on the table" and falls back to the slots, where the
    // nearest noun wins.
    // The stack holds the subject of each clause, outermost first. The main verb agrees with the
    // outermost subject, so read the bottom of the stack. Reading the top would make "the coach
    // that the women who the men saw found" agree with "women".
    if (!head_stack.empty()) return head_stack.front();
    return last_noun;
  };
  std::vector<float> freq_bit(N, 0.0f);
  std::vector<float> bits(N, 0.0f);
  std::vector<int> hist(max_off + 1, UNK);
  std::vector<uint32_t> ctx_idx;
  std::vector<uint32_t> ctx_idx_all;
  int hpos = 0, hfill = 0;

  // stateful context: every bit keeps a decaying trace of how recently it was active
  std::vector<float> trace(N, 0.0f);
  std::vector<std::pair<float, uint32_t>> ranked_trace(N);
  auto update_trace = [&](int v) {
    for (float& t : trace) t *= gamma;
    for (uint32_t j : bits_of(v)) trace[j] += 1.0f;
  };
  auto reset_trace = [&]() { std::fill(trace.begin(), trace.end(), 0.0f); };

  bm::NgramCache ncache(21);
  bm::NgramCache* ncm = use_cache ? &ncache : nullptr;

  auto push_hist = [&](int v) {
    hist[hpos] = v;
    hpos = (hpos + 1) % static_cast<int>(hist.size());
    if (hfill < static_cast<int>(hist.size())) ++hfill;
  };
  auto hist_at = [&](int back) {  // back = 1 is the previous token
    const int idx = (hpos - back + static_cast<int>(hist.size())) % static_cast<int>(hist.size());
    return hist[idx];
  };

  // returns the context bit indices the matrix should read from
  auto context_indices = [&](std::vector<uint32_t>& out) {
    out.clear();
    // only ctx_mode=1 is stateful; modes 0 and 2 both read the fixed slots, so the test must be
    // "is it the trace mode" and not "is it mode 0", or mode 2 silently scores against an
    // all-zero trace and reduces to a constant context
    if (ctx_mode != 1) {
      for (size_t s = 0; s < n_slot; ++s) {
        const int ct = hist_at(offsets[s]);
        if (ct == UNK) continue;
        for (uint32_t i : bits_of(ct)) out.push_back(i);
      }
      return out;
    }
    for (size_t j = 0; j < N; ++j) ranked_trace[j] = {trace[j], static_cast<uint32_t>(j)};
    const size_t take = std::min<size_t>(static_cast<size_t>(trace_k), N);
    std::partial_sort(ranked_trace.begin(), ranked_trace.begin() + take, ranked_trace.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < take; ++i) out.push_back(ranked_trace[i].second);
    return out;
  };

  // score every bit of the output space for a given context
  // returns a reference: returning by value would hand the caller a pointer into a temporary
  auto score_bits = [&]() -> const std::vector<float>& {
    std::fill(bits.begin(), bits.end(), 0.0f);
    ctx_idx.clear();
    context_indices(ctx_idx);
    // When the head is known the slots are the source of the error: they carry the nearest noun,
    // which is the distractor. Down-weight them so the head can win.
    const bool have_head = head_mode && head_bits() != UNK;
    const float slot_w = have_head ? pos_w * slot_head_w : pos_w;
    if (ctx_mode == 1) {
      const float* row = w.data();
      for (uint32_t i : ctx_idx) {
        const float* r = row + static_cast<size_t>(i) * N;
        for (size_t j = 0; j < N; ++j) bits[j] += r[j] * slot_w;
      }
      return bits;
    }
    for (size_t s = 0; s < n_slot; ++s) {
      const int tok = hist_at(offsets[s]);
      if (tok == UNK) continue;
      const float* row = w.data() + s * N * N;
      for (uint32_t i : bits_of(tok)) {
        const float* r = row + static_cast<size_t>(i) * N;
        for (size_t j = 0; j < N; ++j) bits[j] += r[j] * slot_w;
      }
    }
    if (ctx_mode == 2) {
      const float* row = w.data() + bag * N * N;
      for (uint32_t i : ctx_idx) {
        const float* r = row + static_cast<size_t>(i) * N;
        for (size_t j = 0; j < N; ++j) bits[j] += r[j] * slot_w;
      }
    }
    return bits;
  };

  // Per-output-bit mean over context rows. Without it a word whose bits happen to be globally
  // high wins regardless of context, which is exactly what a frequency bias looks like: the model
  // then prefers "turns" after both "keys" and "key".
  std::vector<float> colmean(N, 0.0f);
  auto compute_colmean = [&]() {
    std::fill(colmean.begin(), colmean.end(), 0.0f);
    const size_t rows = (ctx_mode == 1) ? 1 : n_slot;
    const double denom = static_cast<double>(rows * N);
    for (size_t r = 0; r < rows; ++r) {
      const float* row = w.data() + r * N * N;
      for (size_t i = 0; i < N; ++i) {
        const float* rr = row + i * N;
        for (size_t j = 0; j < N; ++j) colmean[j] += rr[j];
      }
    }
    for (size_t j = 0; j < N; ++j) colmean[j] = static_cast<float>(colmean[j] / denom);
  };

  auto word_score = [&](const float* b, int v) {
    if (v == UNK) return 0.0f;
    const float ctx_n = static_cast<float>(ctx_idx.size());
    float sc = 0.0f;
    const bool feat_only = std::getenv("GRAMMAR_FEATONLY") != nullptr;
    for (uint32_t j : bits_of(v)) {
      if (feat_only && j < kFeatBase) continue;
      const float centre = std::getenv("GRAMMAR_NOMEAN") ? 0.0f : colmean[j] * ctx_n;
      // feature bits are shared by many words on purpose, so the frequency normaliser would
      // suppress exactly the signal the hybrid code is meant to expose
      sc += ((code_mode == 2 || code_mode == 4) && !std::getenv("GRAMMAR_NOMEAN"))
                ? (b[j] - centre)
                : (b[j] - centre) / (freq_bit[j] + freq_floor);
    }
    // average, not sum: a word with two features has twice the bits of a word with one, so a sum
    // rewards the longer code regardless of what the context says
    return sc / static_cast<float>(bits_of(v).size() + 1);
  };

  // Competitive readout. The failure in D52 is a global bias: every candidate carries ~7 points of
  // per-bit offset that has nothing to do with the context, so it cancels only if the baseline is
  // taken over the union of the candidates' bits rather than over all columns. With two candidates
  // the centred form is the linear-discriminant score, and it is invariant to any offset shared by
  // the two bit sets.
  auto head_score = [&](int v) {
    if (!head_mode || v == UNK) return 0.0f;
    const int hv = head_bits();
    if (hv == UNK) return 0.0f;
    float sc = 0.0f;
    for (uint32_t i : bits_of(hv)) {
      const float* r = w_head.data() + static_cast<size_t>(i) * N;
      for (uint32_t j : bits_of(v)) sc += r[j];
    }
    return head_w * sc / static_cast<float>(bits_of(v).size() + 1);
  };

  auto pair_scores = [&](const float* b, int gv, int bv) {
    float gs = 0.0f, bs = 0.0f;
    if (readout_mode == 0) {
      gs = word_score(b, gv) + head_score(gv);
      bs = word_score(b, bv) + head_score(bv);
      return std::pair<float, float>{gs, bs};
    }
    std::vector<uint32_t> uni;
    for (uint32_t j : bits_of(gv)) uni.push_back(j);
    for (uint32_t j : bits_of(bv)) uni.push_back(j);
    std::sort(uni.begin(), uni.end());
    uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
    double ub = 0.0;
    for (uint32_t j : uni) ub += b[j];
    ub = uni.empty() ? 0.0 : ub / static_cast<double>(uni.size());
    const float centre = static_cast<float>(ub);
    auto score_c = [&](int v) {
      const auto& bits = bits_of(v);
      float sc = 0.0f;
      for (uint32_t j : bits) sc += b[j] - centre;
      return sc / static_cast<float>(bits.size() + 1);
    };
    gs = score_c(gv) + head_score(gv);
    bs = score_c(bv) + head_score(bv);
    if (readout_mode == 2) {
      // softmax over the two candidates: the comparison becomes a probability, so the caller can
      // report confidence as well as a choice
      const float T = readout_temp;
      const float mx = std::max(gs, bs);
      const float eg = std::exp((gs - mx) / T), eb = std::exp((bs - mx) / T);
      const float pg = eg / (eg + eb);
      gs = std::log(pg + 1e-9f);
      bs = std::log(1.0f - pg + 1e-9f);
    }
    return std::pair<float, float>{gs, bs};
  };

  auto cache_score = [&](int v) {
    if (ncm == nullptr) return 0.0f;
    uint16_t cv[bm::NgramCache::kMaxCont];
    uint8_t cc[bm::NgramCache::kMaxCont];
    uint32_t ids[8];
    int k = 0;
    for (int b = 3; b >= 1 && hfill >= b; --b) ids[k++] = static_cast<uint32_t>(hist_at(b));
    if (k == 0) return 0.0f;
    const int n = ncm->get(bm::NgramCache::hash_context(ids, static_cast<size_t>(k)), cv, cc);
    for (int c = 0; c < n; ++c) {
      if (cv[c] == v) return cache_beta * std::log1p(static_cast<float>(cc[c]));
    }
    return 0.0f;
  };

  // one pass of online training
  std::ifstream in2(path);
  long seen = 0;
  const double t_train0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  while (std::getline(in2, line)) {
    auto toks = Tokenize(line);
    if (toks.size() < 3) continue;
    head_stack.clear();
    last_noun = UNK;
    in_pp = false;
    for (const auto& tok : toks) {
      if (seen >= max_tokens) break;
      const int target = code_of(tok);
      push_head(target);
      if ((ctx_mode == 0 ? hfill >= 1 : true) && target != UNK) {
        ctx_idx_all.clear();
        if (ctx_mode == 2) {
          for (size_t s = 0; s < n_slot; ++s) {
            const int tok_u = hist_at(offsets[s]);
            if (tok_u == UNK) continue;
            for (uint32_t i : bits_of(tok_u)) ctx_idx_all.push_back(i);
          }
        }
        for (size_t s = 0; s < n_slot; ++s) {
          // in trace mode there is one matrix, not one per slot: the context is a state, and a
          // state has no position
          const size_t slot = (ctx_mode == 1) ? 0 : s;
          float* row = w.data() + slot * N * N;
          // Each slot must see only its own word. Feeding every slot the union of all context bits
          // makes the slots identical by construction: position then carries no information and the
          // model degenerates into a bag of words, which is why truncating a relative clause changed
          // nothing at evaluation time.
          ctx_idx.clear();
          if (ctx_mode == 1) {
            context_indices(ctx_idx);
          } else {
            const int tok = hist_at(offsets[s]);
            if (tok != UNK) {
              for (uint32_t i : bits_of(tok)) ctx_idx.push_back(i);
            }
          }
          if (ctx_mode == 2 && pos_w <= 0.0f) continue;
          for (uint32_t i : ctx_idx) {
            float* r = row + static_cast<size_t>(i) * N;
            for (uint32_t j : bits_of(target)) {
              // importance-scaled: the synapse slows down as it accumulates change, which is
              // what keeps a pair that has been seen 50k times from pinning the matrix
              float& om = omega[slot * N * N + static_cast<size_t>(i) * N + j];
              const float xi_eff = (j >= kFeatBase) ? xi * feat_boost : xi;
              const float dw = (xi_eff / (1.0f + lam * om)) * (1.0f - r[j]);
              r[j] += dw;
              om += std::fabs(dw);
            }
          }
          if (ctx_mode == 1) break;
        }
        if (ctx_mode == 2) {
          const float* row = w.data() + bag * N * N;
          for (uint32_t i : ctx_idx_all) {
            float* r = const_cast<float*>(row) + static_cast<size_t>(i) * N;
            for (uint32_t j : bits_of(target)) {
              float& om = omega[bag * N * N + static_cast<size_t>(i) * N + j];
              const float dw = (xi / (1.0f + lam * om)) * (1.0f - r[j]);
              r[j] += dw;
              om += std::fabs(dw);
            }
          }
        }
        if (head_mode && head_bits() != UNK) {
          const int hv = head_bits();
          if (hv != UNK) {
            ++g_head_updates;
            if (hv == code_of("one")) ++g_head_one;
            for (uint32_t i : bits_of(hv)) {
              float* r = w_head.data() + static_cast<size_t>(i) * N;
              for (uint32_t j : bits_of(target)) {
                float& om = omega_head[static_cast<size_t>(i) * N + j];
                const float dw = (xi / (1.0f + lam * om)) * (1.0f - r[j]);
                r[j] += dw;
                om += std::fabs(dw);
              }
            }
          }
        }
        for (uint32_t j : bits_of(target)) {
          freq_bit[j] += xi * (1.0f - freq_bit[j]);
        }
        // synaptic homeostasis (SHY): every 400k tokens, the weakest 80% shrink by 18%
        if (seen % 400000 == 0) {
          float lo = w[0], hi = w[0];
          for (float v : w) {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
          }
          float thr = lo;
          if (hi > lo) {
            const int B = 256;
            std::vector<int> histB(B, 0);
            const float span = hi - lo;
            for (float v : w) histB[static_cast<int>((v - lo) / span * (B - 1))]++;
            const long want = static_cast<long>(0.8 * static_cast<double>(w.size()));
            long acc = 0;
            for (int b = 0; b < B; ++b) {
              acc += histB[b];
              if (acc >= want) {
                thr = lo + span * static_cast<float>(b) / static_cast<float>(B - 1);
                break;
              }
            }
          }
          for (float& v : w) {
            if (v < thr) v *= 0.82f;
          }
        }
        if (ncm != nullptr && hfill >= 3) {
          uint32_t ids[3];
          for (int b = 0; b < 3; ++b) ids[b] = static_cast<uint32_t>(hist_at(3 - b));
          ncm->put(bm::NgramCache::hash_context(ids, 3), static_cast<uint16_t>(target));
          if (ncm->load() > 0.9) ncm->sweep(0.15);
        }
      }
      if (ctx_mode == 1) update_trace(target);
      push_hist(target);
      ++seen;
    }
    if (seen >= max_tokens) break;
  }
  const double t_train = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - t_train0;
  printf("ctx_mode=%d code_mode=%d slots=%zu trace_k=%d gamma=%.2f\n", ctx_mode, code_mode,
         n_slot, trace_k, gamma);
  printf("trained_tokens=%ld train_seconds=%.1f cache_live=%zu cache_bytes=%.1fMB\n", seen, t_train,
         ncm ? ncm->used() : 0, static_cast<double>(ncm ? ncm->bytes() : 0) / (1024.0 * 1024.0));

  // forced-choice evaluation
  struct Bucket {
    long n = 0, ok = 0;
  };
  std::vector<std::pair<std::string, Bucket>> buckets;
  auto bucket = [&](const std::string& name) -> Bucket& {
    for (auto& b : buckets) {
      if (b.first == name) return b.second;
    }
    buckets.emplace_back(name, Bucket{});
    return buckets.back().second;
  };

  compute_colmean();
  if (std::getenv("GRAMMAR_BLOCKS")) {
    auto block = [&](std::vector<Feat> a, std::vector<Feat> b) {
      double sum = 0.0;
      int n = 0;
      for (Feat fa : a) {
        for (Feat fb : b) {
          for (int ra = 0; ra < feat_repeat; ++ra) {
            for (int rb = 0; rb < feat_repeat; ++rb) {
              const size_t i = kFeatBase + static_cast<size_t>(fa) * feat_repeat + ra;
              const size_t j = kFeatBase + static_cast<size_t>(fb) * feat_repeat + rb;
              sum += w[i * N + j];
              ++n;
            }
          }
        }
      }
      return sum / n;
    };
    printf("blocks PLxBASE=%.4f PLx3SG=%.4f SGx3SG=%.4f SGxBASE=%.4f\n",
           block({F_PLURAL}, {F_VERBBASE}), block({F_PLURAL}, {F_VERB3SG}),
           block({F_SINGULAR}, {F_VERB3SG}), block({F_SINGULAR}, {F_VERBBASE}));
    // per-slot blocks: if the long-distance agreement exists at all it must live in the slot that
    // corresponds to the subject's distance, not in slot 0
    for (size_t s = 0; s < n_slot; ++s) {
      const float* row = w.data() + s * N * N;
      auto blk = [&](std::vector<Feat> a, std::vector<Feat> b) {
        double sum = 0.0;
        int n = 0;
        for (Feat fa : a) {
          for (Feat fb : b) {
            for (int ra = 0; ra < feat_repeat; ++ra) {
              for (int rb = 0; rb < feat_repeat; ++rb) {
                const size_t i = kFeatBase + static_cast<size_t>(fa) * feat_repeat + ra;
                const size_t j = kFeatBase + static_cast<size_t>(fb) * feat_repeat + rb;
                sum += row[i * N + j];
                ++n;
              }
            }
          }
        }
        return sum / n;
      };
      printf("slot%zu PLxBASE=%.4f PLx3SG=%.4f SGx3SG=%.4f SGxBASE=%.4f\n", s,
             blk({F_PLURAL}, {F_VERBBASE}), blk({F_PLURAL}, {F_VERB3SG}),
             blk({F_SINGULAR}, {F_VERB3SG}), blk({F_SINGULAR}, {F_VERBBASE}));
    }
  }
  long covered = 0, total = 0, correct = 0;
  long covered_u = 0, correct_u = 0, covered_a = 0, correct_a = 0;
  for (const auto& pr : pairs) {
    ++total;
    const int gv = code_of(pr.good);
    const int bv = code_of(pr.bad);
    // A pair whose context contains a word the model has never seen cannot be scored: that word
    // becomes UNK, loses its feature, and the item degenerates into a coin flip that the global
    // bias decides. Counting it would inflate the error rate for a reason that has nothing to do
    // with grammar.
    bool ctx_unknown = false;
    for (const auto& w : pr.ctx) {
      if (code_of(w) == UNK) ctx_unknown = true;
    }
    if (gv == UNK || bv == UNK || ctx_unknown) {
      // not counted at all: a pair the model cannot represent would drag the per-phenomenon
      // accuracy down for a reason that has nothing to do with grammar
      continue;
    }
    if (ctx_mode == 1) reset_trace();
    // ctx_mode=2: truncate the context at the first relativiser. This is the access experiment. If
    // the model then answers the pairs it previously got wrong, the agreement rule is intact and
    // the only failure is that a bag of features cannot tell which noun is the subject.
    size_t ctx_len = pr.ctx.size();
    if (ctx_mode == 3) {
      for (size_t i = 1; i < pr.ctx.size(); ++i) {
        // i starts at 1: "that" in clause-initial position is a determiner, not a relativiser
        if (pr.ctx[i] == "that" || pr.ctx[i] == "who" || pr.ctx[i] == "which") {
          ctx_len = i;
          break;
        }
      }
    }
    if (ctx_mode == 1) reset_trace();
    head_stack.clear();
    last_noun = UNK;
    in_pp = false;
    for (size_t i = 0; i < ctx_len; ++i) {
      const int cv = code_of(pr.ctx[i]);
      push_hist(cv);
      push_head(cv);
      if (ctx_mode == 1 && cv != UNK) update_trace(cv);
    }
    // contexts shorter than the model wants are padded with UNK, which is what happens at a
    // sentence start; the pair is still scorable
    const float* b = score_bits().data();
    const auto ps = pair_scores(b, gv, bv);
    const float gs = ps.first + cache_score(gv);
    const float bs = ps.second + cache_score(bv);
    if (std::getenv("GRAMMAR_DEBUG") != nullptr) {
      printf("DBG head=%d headscore=%.4f/%.4f good=%s(%.6f) bad=%s(%.6f) ctx=%s\n", head_bits(),
             head_score(gv), head_score(bv), pr.good.c_str(), gs, pr.bad.c_str(), bs,
             [&] {
               std::string o;
               for (const auto& w : pr.ctx) o += w + " ";
               return o;
             }()
                 .c_str());
    }
    const bool ok = gs > bs;
    if (ok) ++correct;
    ++covered;
    if (pr.amb) {
      ++covered_a;
      correct_a += ok ? 1 : 0;
    } else {
      ++covered_u;
      correct_u += ok ? 1 : 0;
    }
    bucket(pr.phenomenon).n++;
    bucket(pr.phenomenon).ok += ok ? 1 : 0;
    // reset history and trace so items do not contaminate each other
    hpos = 0;
    hfill = 0;
    if (ctx_mode == 1) reset_trace();
  }

  printf("\n%-22s %6s %6s %7s\n", "phenomenon", "n", "ok", "acc");
  for (const auto& b : buckets) {
    const double acc = b.second.n ? static_cast<double>(b.second.ok) / b.second.n : 0.0;
    printf("%-22s %6ld %6ld %6.1f%%\n", b.first.c_str(), b.second.n, b.second.ok, acc * 100.0);
  }
  if (covered_u > 0) {
    printf("%-22s %6ld %6ld %6.1f%%\n", "unambiguous", covered_u, correct_u,
           100.0 * static_cast<double>(correct_u) / covered_u);
  }
  if (covered_a > 0) {
    printf("%-22s %6ld %6ld %6.1f%%\n", "ambiguous", covered_a, correct_a,
           100.0 * static_cast<double>(correct_a) / covered_a);
  }
  printf("%-22s %6ld %6ld %6.1f%%  (chance 50%%)\n", "OVERALL", covered, correct,
         covered ? 100.0 * correct / covered : 0.0);
  printf("pairs_total=%ld scorable=%ld not_in_vocab=%ld head_updates=%ld push_one=%ld head_one=%ld\n",
         total, covered, total - covered, g_head_updates, g_push_one, g_head_one);
  return 0;
}
