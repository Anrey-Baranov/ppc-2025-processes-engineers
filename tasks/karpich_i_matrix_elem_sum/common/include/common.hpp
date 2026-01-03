#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace karpich_i_matrix_elem_sum {

using InType = std::tuple<std::size_t, std::size_t, std::vector<int>>;
using OutType = std::int64_t;
using TestType = std::tuple<std::string, std::int64_t>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace karpich_i_matrix_elem_sum
