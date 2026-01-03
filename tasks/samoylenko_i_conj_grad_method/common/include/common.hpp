#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace samoylenko_i_conj_grad_method {

using InType = std::pair<int, int>;
using OutType = std::vector<double>;
using TestType = std::tuple<std::pair<int, int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace samoylenko_i_conj_grad_method
