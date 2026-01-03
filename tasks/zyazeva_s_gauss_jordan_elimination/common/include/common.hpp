#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

using InType = std::vector<std::vector<float>>;
using OutType = std::vector<float>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zyazeva_s_gauss_jordan_elimination
