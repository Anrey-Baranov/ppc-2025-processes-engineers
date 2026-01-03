#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kolotukhin_a_hypercube {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<std::vector<int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kolotukhin_a_hypercube
