#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace trofimov_n_linear_topology {

struct InputData {
  int source;
  int target;
  int value;
};

inline void Work(int n) {
  volatile int acc = 0;
  const int iters = n * 1000;
  for (int i = 0; i < iters; ++i) {
    acc += i % 13;
    acc ^= acc << 1;
    acc += acc >> 3;
  }
}

using InType = InputData;
using OutType = int;

using TestType = std::tuple<InputData, std::string>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace trofimov_n_linear_topology
