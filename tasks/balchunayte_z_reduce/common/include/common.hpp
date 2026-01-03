#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace balchunayte_z_reduce {

struct InType {
  std::vector<double> data;
  int root;
};

using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace balchunayte_z_reduce
