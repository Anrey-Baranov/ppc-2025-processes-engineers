#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace yakimov_i_linear_virtual_topology {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace yakimov_i_linear_virtual_topology
