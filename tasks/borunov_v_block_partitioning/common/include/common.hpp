#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace borunov_v_block_partitioning {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace borunov_v_block_partitioning
