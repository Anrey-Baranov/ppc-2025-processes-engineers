#pragma once

#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace tochilin_e_integration_trapezoid {

using FunctionType = std::function<double(double)>;

struct IntegrationInput {
  double lower_bound{};
  double upper_bound{};
  int num_intervals{};
  FunctionType function;
};

using InType = IntegrationInput;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tochilin_e_integration_trapezoid
