#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace afanasyev_a_it_seidel_method {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace afanasyev_a_it_seidel_method
