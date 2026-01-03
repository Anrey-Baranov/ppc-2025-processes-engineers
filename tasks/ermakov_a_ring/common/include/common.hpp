#pragma once

#include <vector>

#include "task/include/task.hpp"

namespace ermakov_a_ring {

struct RingTaskData {
  int source = 0;
  int dest = 0;
  std::vector<int> data;
};

using InType = RingTaskData;
using OutType = std::vector<int>;
using TestType = int;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ermakov_a_ring
