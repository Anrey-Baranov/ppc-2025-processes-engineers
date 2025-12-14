#include "baranov_a_dijkstra_crs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <variant>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"

namespace baranov_a_dijkstra_crs {

BaranovADijkstraCrsSEQ::BaranovADijkstraCrsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BaranovADijkstraCrsSEQ::ValidationImpl() {
  std::cout << "[DEBUG] BaranovADijkstraCrsSEQ::ValidationImpl called" << std::endl;
  try {
    if (!std::holds_alternative<GraphCRS>(GetInput())) {
      return false;
    }

    const auto &graph = std::get<GraphCRS>(GetInput());

    if (graph.vertices <= 0) {
      return false;
    }
    if (graph.source < 0 || graph.source >= graph.vertices) {
      return false;
    }
    if (graph.row_ptr.size() != static_cast<size_t>(graph.vertices) + 1) {
      return false;
    }

    return true;

  } catch (...) {
    return false;
  }
}

bool BaranovADijkstraCrsSEQ::PreProcessingImpl() {
  std::cout << "[DEBUG] BaranovADijkstraCrsSEQ::PreProcessingImpl called" << std::endl;
  return true;
}

template <typename T>
std::vector<T> BaranovADijkstraCrsSEQ::DijkstraSequentialTemplate(int vertices, const std::vector<int> &row_ptr,
                                                                  const std::vector<int> &col_idx,
                                                                  const std::vector<T> &values, int source) {
  const T INF = std::numeric_limits<T>::max();
  std::vector<T> dist(vertices, INF);
  std::vector<bool> visited(vertices, false);

  dist[source] = static_cast<T>(0);

  for (int i = 0; i < vertices; ++i) {
    int u = -1;
    T min_dist = INF;

    for (int v = 0; v < vertices; ++v) {
      if (!visited[v] && dist[v] < min_dist) {
        min_dist = dist[v];
        u = v;
      }
    }

    if (u == -1 || dist[u] == INF) {
      break;
    }

    visited[u] = true;
    int start = row_ptr[u];
    int end = row_ptr[u + 1];

    for (int j = start; j < end; ++j) {
      int v = col_idx[j];
      T weight = values[j];

      if (!visited[v] && dist[u] + weight < dist[v]) {
        dist[v] = dist[u] + weight;
      }
    }
  }

  return dist;
}

bool BaranovADijkstraCrsSEQ::RunImpl() {
  std::cout << "[DEBUG] BaranovADijkstraCrsSEQ::RunImpl called" << std::endl;
  try {
    const auto &graph = std::get<GraphCRS>(GetInput());

    if (std::holds_alternative<int>(graph.weights)) {
      int weight = std::get<int>(graph.weights);
      std::vector<int> values(graph.col_idx.size(), weight);
      auto result = DijkstraSequentialTemplate<int>(graph.vertices, graph.row_ptr, graph.col_idx, values, graph.source);
      GetOutput() = result;
    } else if (std::holds_alternative<float>(graph.weights)) {
      float weight = std::get<float>(graph.weights);
      std::vector<float> values(graph.col_idx.size(), weight);
      auto result =
          DijkstraSequentialTemplate<float>(graph.vertices, graph.row_ptr, graph.col_idx, values, graph.source);
      GetOutput() = result;
    } else if (std::holds_alternative<double>(graph.weights)) {
      double weight = std::get<double>(graph.weights);
      std::vector<double> values(graph.col_idx.size(), weight);
      auto result =
          DijkstraSequentialTemplate<double>(graph.vertices, graph.row_ptr, graph.col_idx, values, graph.source);
      GetOutput() = result;
    } else {
      return false;
    }

    return true;

  } catch (...) {
    return false;
  }
}

bool BaranovADijkstraCrsSEQ::PostProcessingImpl() {
  std::cout << "[DEBUG] BaranovADijkstraCrsSEQ::PostProcessingImpl called" << std::endl;
  return true;
}

}  // namespace baranov_a_dijkstra_crs
