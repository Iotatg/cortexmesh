// File-level retrieval for real GitHub issues (SWE-bench Lite).
//
// Task: given an issue statement, retrieve the source file that the gold patch edits.
// There is no training: each issue is hashed into a sparse code (word hashing), the same is
// done for the candidate files (identified by path + the files' own text is NOT available
// here, so candidates are scored by path-token overlap only), and the ranking is by
// overlap. Baselines: random ranking and shuffled labels.
//
// Usage: ./bin/swebench <tsv> [bits] [per_word]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

uint64_t Mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

struct Doc {
  std::string id;
  std::vector<std::string> files;
  std::vector<std::string> tokens;
};

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  std::stringstream ss(s);
  while (std::getline(ss, cur, sep)) out.push_back(cur);
  return out;
}

std::vector<std::string> Tokenize(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '/') {
      cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    } else {
      if (cur.size() > 1) out.push_back(cur);
      cur.clear();
    }
  }
  if (cur.size() > 1) out.push_back(cur);
  return out;
}

std::vector<uint64_t> HashCode(const std::vector<std::string>& toks, size_t bits, size_t per_word) {
  std::vector<uint64_t> code(bits, 0);
  for (const auto& t : toks) {
    for (size_t r = 0; r < per_word; ++r) {
      uint64_t h = Mix(std::hash<std::string>{}(t) ^ (0x1000193ULL * (r + 1)));
      code[h % bits] = 1;
    }
  }
  return code;
}

double Similarity(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
  size_t inter = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    inter += static_cast<size_t>(a[i] & b[i]);
    na += static_cast<size_t>(a[i]);
    nb += static_cast<size_t>(b[i]);
  }
  if (na + nb == 0) return 0.0;
  return 2.0 * double(inter) / double(na + nb);  // Dice
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "data/swebench_lite.tsv";
  const size_t bits = argc > 2 ? static_cast<size_t>(std::strtoul(argv[2], nullptr, 10)) : 1024;
  const size_t per_word = argc > 3 ? static_cast<size_t>(std::strtoul(argv[3], nullptr, 10)) : 2;

  std::ifstream in(path);
  if (!in) {
    printf("cannot open %s\n", path.c_str());
    return 1;
  }

  std::vector<Doc> docs;
  std::string line;
  while (std::getline(in, line)) {
    auto f = Split(line, '\t');
    if (f.size() < 4) continue;
    Doc d;
    d.id = f[0];
    d.files = Split(f[2], ',');
    d.tokens = Tokenize(f[3]);
    docs.push_back(std::move(d));
  }
  printf("instances=%zu bits=%zu per_word=%zu\n", docs.size(), bits, per_word);

  // candidate universe: every file path mentioned by any gold patch
  std::unordered_map<std::string, int> cand_index;
  std::vector<std::string> cands;
  for (const auto& d : docs) {
    for (const auto& f : d.files) {
      if (!cand_index.count(f)) {
        cand_index[f] = static_cast<int>(cands.size());
        cands.push_back(f);
      }
    }
  }
  printf("candidate_files=%zu\n", cands.size());

  std::vector<std::vector<uint64_t>> cand_code(cands.size());
  for (size_t i = 0; i < cands.size(); ++i) cand_code[i] = HashCode(Tokenize(cands[i]), bits, per_word);

  long h1 = 0, h5 = 0, h10 = 0, n = 0;
  double mrr = 0.0;
  long rand_h1 = 0, rand_h5 = 0;
  std::mt19937_64 rng(7);
  for (const auto& d : docs) {
    auto q = HashCode(d.tokens, bits, per_word);
    std::vector<std::pair<double, int>> scored;
    scored.reserve(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) scored.emplace_back(Similarity(q, cand_code[i]), static_cast<int>(i));
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> gold;
    for (const auto& f : d.files) {
      auto it = cand_index.find(f);
      if (it != cand_index.end()) gold.push_back(it->second);
    }
    if (gold.empty()) continue;
    ++n;
    for (size_t r = 0; r < scored.size(); ++r) {
      if (std::find(gold.begin(), gold.end(), scored[r].second) != gold.end()) {
        h1 += (r == 0);
        h5 += (r < 5);
        h10 += (r < 10);
        mrr += 1.0 / double(r + 1);
        break;
      }
    }
    // random control on the same instance
    std::vector<int> perm(cands.size());
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<int>(i);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (size_t r = 0; r < perm.size(); ++r) {
      if (std::find(gold.begin(), gold.end(), perm[r]) != gold.end()) {
        rand_h1 += (r == 0);
        rand_h5 += (r < 5);
        break;
      }
    }
  }
  printf("retrieval,n=%ld,top1=%.4f,top5=%.4f,top10=%.4f,mrr=%.4f\n", n, double(h1) / n,
         double(h5) / n, double(h10) / n, mrr / n);
  printf("random,n=%ld,top1=%.4f,top5=%.4f\n", n, double(rand_h1) / n, double(rand_h5) / n);
  return 0;
}