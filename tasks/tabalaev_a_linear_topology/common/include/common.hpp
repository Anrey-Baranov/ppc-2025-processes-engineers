#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tabalaev_a_linear_topology {

using InType = std::tuple<int, int, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tabalaev_a_linear_topology
