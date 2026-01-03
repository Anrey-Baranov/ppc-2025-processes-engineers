#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace bortsova_a_transmission_gather {

struct Input {
  std::vector<double> send_data;
  int root = 0;
};

struct Output {
  std::vector<double> recv_data;
};

using InType = Input;
using OutType = Output;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace bortsova_a_transmission_gather
