#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace vasiliev_m_bellman_ford_crs {

using WeightType = int;  // веса ребер - целое число

struct CRSGraph {
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<WeightType> vals;
  int source = 0;
};

using InType = CRSGraph;
using OutType = std::vector<WeightType>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace vasiliev_m_bellman_ford_crs
