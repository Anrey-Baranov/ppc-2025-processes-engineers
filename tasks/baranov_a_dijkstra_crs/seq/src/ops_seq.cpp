#include "baranov_a_dijkstra_crs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"

namespace baranov_a_dijkstra_crs {

BaranovADijkstraCRSSEQ::BaranovADijkstraCRSSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool BaranovADijkstraCRSSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.num_vertices <= 0) {
    return false;
  }
  if (input.source_vertex < 0 || input.source_vertex >= input.num_vertices) {
    return false;
  }
  if (input.offsets.size() != static_cast<std::size_t>(input.num_vertices + 1)) {
    return false;
  }
  return true;
}

bool BaranovADijkstraCRSSEQ::PreProcessingImpl() {
  return true;
}

bool BaranovADijkstraCRSSEQ::RunImpl() {
  const auto &graph = GetInput();
  const int n = graph.num_vertices;
  const int source = graph.source_vertex;
  std::vector<double> dist(n, std::numeric_limits<double>::infinity());
  dist[source] = 0.0;

  std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> pq;
  pq.emplace(0.0, source);

  std::vector<bool> visited(n, false);

  while (!pq.empty()) {
    auto [current_dist, u] = pq.top();
    pq.pop();

    if (visited[u]) {
      continue;
    }
    visited[u] = true;

    int start = graph.offsets[u];
    int end = graph.offsets[u + 1];

    for (int idx = start; idx < end; ++idx) {
      int v = graph.columns[idx];
      double weight = graph.values[idx];

      if (!visited[v]) {
        double new_dist = current_dist + weight;
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
          pq.emplace(new_dist, v);
        }
      }
    }
  }

  GetOutput() = dist;
  return true;
}

bool BaranovADijkstraCRSSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace baranov_a_dijkstra_crs
