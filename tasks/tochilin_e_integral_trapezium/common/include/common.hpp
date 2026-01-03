#pragma once

#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tochilin_e_integral_trapezium {

struct IntegralInput {
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
  int num_steps{0};
  std::function<double(const std::vector<double> &)> func;
};

using InType = IntegralInput;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tochilin_e_integral_trapezium
