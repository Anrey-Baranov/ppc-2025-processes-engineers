#pragma once

#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "task/include/task.hpp"

namespace sinev_a_allreduce {

using InTypeVariant = std::variant<std::vector<int>, std::vector<float>, std::vector<double>>;

using InType = InTypeVariant;
using OutType = InTypeVariant;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sinev_a_allreduce
