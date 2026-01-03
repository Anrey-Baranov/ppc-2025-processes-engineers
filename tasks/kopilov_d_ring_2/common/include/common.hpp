#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kopilov_d_ring_2 {

struct Input {
  std::vector<int> data;
};

struct Output {
  std::vector<int> data;
};

using InType = Input;
using OutType = Output;

using TestType = std::tuple<std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kopilov_d_ring_2
