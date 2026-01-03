#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tochilin_e_vertical_ribbon_scheme {

struct TaskData {
  std::vector<double> matrix;
  std::vector<double> vector;
  int rows{0};
  int cols{0};
};

using InType = TaskData;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tochilin_e_vertical_ribbon_scheme
