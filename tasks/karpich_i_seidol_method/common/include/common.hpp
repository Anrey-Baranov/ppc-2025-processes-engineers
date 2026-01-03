#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace karpich_i_seidol_method {

using InType = std::tuple<std::size_t, std::vector<double>, std::vector<double>, double>;
using OutType = std::vector<double>;
using TestType = std::tuple<std::string, int, int, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace karpich_i_seidol_method
