#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kolotukhin_a_merge_sort_doubles {

using InType = std::vector<double>;
using OutType = std::tuple<std::vector<double>, int>;
using TestType = std::tuple<std::tuple<std::vector<double>, std::vector<double>>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kolotukhin_a_merge_sort_doubles
