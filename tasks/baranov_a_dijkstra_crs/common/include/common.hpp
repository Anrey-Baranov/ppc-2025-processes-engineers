#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace baranov_a_dijkstra_crs {

struct GraphData {
  std::vector<int> offsets;
  std::vector<int> columns;
  std::vector<double> values;
  int num_vertices = 0;
  int source_vertex = 0;

  GraphData() : num_vertices(0), source_vertex(0) {}
  GraphData(int n, int src) : num_vertices(n), source_vertex(src) {}
};

using InType = GraphData;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace baranov_a_dijkstra_crs
