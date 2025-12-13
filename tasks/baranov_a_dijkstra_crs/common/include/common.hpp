#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace baranov_a_dijkstra_crs {

using WeightTypeVariant = std::variant<int, float, double>;

struct GraphCRS {
  int vertices;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  WeightTypeVariant weights;
  int source;
};

using InTypeVariant = std::variant<GraphCRS, int, std::vector<int>>;

using InType = InTypeVariant;
using OutType = std::variant<std::vector<int>, std::vector<float>, std::vector<double>>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

struct TestConfig {
  int vertices;
  int source;
  std::string weight_type;
  float density;
};

}  // namespace baranov_a_dijkstra_crs
