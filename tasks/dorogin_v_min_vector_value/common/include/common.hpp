#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dorogin_v_min_vector_value {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<int, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dorogin_v_min_vector_value
