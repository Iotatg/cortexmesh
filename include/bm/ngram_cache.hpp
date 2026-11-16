// Hashed n-gram memory with LRU-style aging and a fixed byte budget.
//
// Why this exists (docs/37, D45-D46): the dense Hebbian matrix in tools/streamlm.cpp cannot be
// fixed by any update rule, because every mechanism that removes saturation also removes the
// frequent pairs that carry the signal. The alternative is a memory that does not accumulate:
// a hash table with a hard capacity, where an entry is either present or evicted.
//
// This is a *memory address*, not a representation. Word ids are used only to hash a context into
// a 64-bit key; the meaning of a word still lives in its sparse code (bm::SDR) everywhere else.
//
// Layout: open addressing, linear probing, 16 bytes per slot.
//   key  8B  (0 = empty)
//   val  2B  (word code id that followed this context)
//   cnt  2B  (saturating hit counter, capped at 65535)
//   age  1B  (touch clock, used by the sweep)
//   pad  3B
// 2^21 slots = 32 MB. Eviction is by age, not by insertion order, so a hot context survives while
// a one-off context is dropped even if it was written later.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace bm {

class NgramCache {
 public:
  static constexpr int kMaxCont = 3;

  // 8B key + 3 x (2B value + 1B count) + 1B age = 18B, padded to 24B.
  // The count is per continuation, capped at 255, so a context that has only ever been followed
  // by one word cannot masquerade as a distribution. Without this the table would report
  // top-3 and top-10 identical to top-1, which is a measurement artefact, not a result.
  struct Slot {
    uint64_t key;
    uint16_t val[kMaxCont];
    uint8_t cnt[kMaxCont];
    uint8_t age;
    uint8_t pad[3];
  };

  explicit NgramCache(size_t bits = 21, float fill_limit = 0.85f)
      : mask_((static_cast<size_t>(1) << bits) - 1), slots_(static_cast<size_t>(1) << bits),
        fill_limit_(fill_limit) {}

  size_t capacity() const { return slots_.size(); }
  size_t used() const { return used_; }
  double load() const { return static_cast<double>(used_) / static_cast<double>(slots_.size()); }
  size_t bytes() const { return slots_.size() * sizeof(Slot); }

  // splitmix64: one multiply-xorshift round per word, then a final avalanche.
  static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
  }

  static uint64_t hash_context(const uint32_t* ids, size_t k) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < k; ++i) h = mix(h ^ ids[i]);
    return h == 0 ? 1 : h;
  }

  // Returns false on a miss. On a hit, val and the saturating count are reported.
  // Number of continuations returned in out_val / out_cnt (at most kMaxCont).
  int get(uint64_t key, uint16_t* out_val, uint8_t* out_cnt) const {
    size_t i = static_cast<size_t>(key) & mask_;
    for (size_t probe = 0; probe < 64; ++probe) {
      const Slot& s = slots_[i];
      if (s.key == 0) return 0;
      if (s.key == key) {
        int n = 0;
        for (int c = 0; c < kMaxCont; ++c) {
          if (s.cnt[c] == 0) continue;
          out_val[n] = s.val[c];
          out_cnt[n] = s.cnt[c];
          ++n;
        }
        return n;
      }
      i = (i + 1) & mask_;
    }
    return 0;
  }

  // Insert or reinforce. A hit increments the count; a miss takes the next free slot, or the
  // coldest slot in the probe chain once the table is past the fill limit.
  void put(uint64_t key, uint16_t val) {
    if (key == 0) key = 1;
    const bool full = used_ >= fill_limit_ * slots_.size();
    size_t i = static_cast<size_t>(key) & mask_;
    size_t coldest = i;
    for (size_t probe = 0; probe < 64; ++probe) {
      Slot& s = slots_[i];
      if (s.key == 0) {
        place(full ? coldest : i, key, val);
        if (!full) ++used_;
        return;
      }
      if (s.key == key) {
        for (int c = 0; c < kMaxCont; ++c) {
          if (s.cnt[c] != 0 && s.val[c] == val) {
            if (s.cnt[c] < 255) ++s.cnt[c];
            s.age = tick_;
            return;
          }
        }
        int slot = -1, smallest = 1 << 30;
        for (int c = 0; c < kMaxCont; ++c) {
          if (s.cnt[c] == 0) {
            slot = c;
            break;
          }
          if (s.cnt[c] < smallest) {
            smallest = s.cnt[c];
            slot = c;
          }
        }
        if (slot < 0) slot = 0;
        s.val[slot] = val;
        s.cnt[slot] = 1;
        s.age = tick_;
        return;
      }
      if (s.age < slots_[coldest].age) coldest = i;
      i = (i + 1) & mask_;
    }
    place(coldest, key, val);
  }

  // Called once per sweep period. Drops the coldest fraction, which is what keeps a hot context
  // alive and a one-off context from pinning a slot forever.
  void sweep(float drop_fraction = 0.15f) {
    ++tick_;
    if (used_ < 0.5 * slots_.size()) return;
    const size_t drop = static_cast<size_t>(drop_fraction * slots_.size());
    if (drop == 0) return;
    std::vector<size_t> idx;
    idx.reserve(drop * 4);
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].key != 0) idx.push_back(i);
    }
    if (idx.size() <= drop) {
      for (size_t i : idx) {
        slots_[i].key = 0;
        --used_;
      }
      return;
    }
    std::partial_sort(idx.begin(), idx.begin() + drop, idx.end(),
                      [this](size_t a, size_t b) { return slots_[a].age < slots_[b].age; });
    for (size_t i = 0; i < drop; ++i) {
      slots_[idx[i]].key = 0;
      --used_;
    }
  }

 private:
  void place(size_t i, uint64_t key, uint16_t val) {
    Slot& s = slots_[i];
    s.key = key;
    s.val[0] = val;
    s.cnt[0] = 1;
    s.val[1] = 0;
    s.cnt[1] = 0;
    s.val[2] = 0;
    s.cnt[2] = 0;
    s.age = tick_;
    std::memset(s.pad, 0, sizeof(s.pad));
  }

  size_t mask_;
  std::vector<Slot> slots_;
  size_t used_ = 0;
  float fill_limit_;
  uint8_t tick_ = 0;
};

}  // namespace bm