#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace bmtest {

inline int g_failures = 0;
inline int g_checks = 0;

inline void check(bool cond, const std::string& what, const char* file, int line) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("FAIL %s:%d  %s\n", file, line, what.c_str());
  }
}

inline int summary(const char* name) {
  std::printf("%s: %d checks, %d failures\n", name, g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace bmtest

#define CHECK(cond) bmtest::check((cond), #cond, __FILE__, __LINE__)