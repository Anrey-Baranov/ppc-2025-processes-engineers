#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace redkina_a_ruler {

struct RulerMessage {
  int start{};
  int end{};
  std::vector<int> data;
};

using InType = RulerMessage;
using OutType = std::vector<int>;
using TestType = std::tuple<int, RulerMessage>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace redkina_a_ruler
