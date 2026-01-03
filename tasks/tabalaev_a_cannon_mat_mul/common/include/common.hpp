#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tabalaev_a_cannon_mat_mul {

using InType = std::tuple<size_t, std::vector<double>, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<size_t, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tabalaev_a_cannon_mat_mul
