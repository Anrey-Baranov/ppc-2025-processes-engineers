#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kamaletdinov_r_max_matrix_rows_elem {

using InType = std::tuple<std::size_t, std::size_t, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::string, std::string, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kamaletdinov_r_max_matrix_rows_elem
