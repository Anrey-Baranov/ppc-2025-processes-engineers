#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace zaharov_g_seidel_int_met {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zaharov_g_seidel_int_met
