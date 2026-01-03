#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace buzulukskiy_d_sort_batcher {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace buzulukskiy_d_sort_batcher
