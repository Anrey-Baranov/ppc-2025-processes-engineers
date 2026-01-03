#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace timofeev_n_radix_merge_sort {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::vector<int>, OutType, InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace timofeev_n_radix_merge_sort
