#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace makoveeva_s_cannon_algorithm {

using InType = std::tuple<std::vector<double>, std::vector<double>, int>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace makoveeva_s_cannon_algorithm
