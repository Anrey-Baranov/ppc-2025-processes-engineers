#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kondrashova_v_ring_topology {

struct Data {
  int source = 0;
  int recipient = 0;
  std::vector<int> data;
};

using InType = Data;
using OutType = std::vector<int>;

using TestType = std::tuple<int, int, std::vector<int>, std::string>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kondrashova_v_ring_topology
