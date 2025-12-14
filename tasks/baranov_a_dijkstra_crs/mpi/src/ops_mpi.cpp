#include "baranov_a_dijkstra_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <variant>
#include <vector>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"

namespace baranov_a_dijkstra_crs {

BaranovADijkstraCrsMPI::BaranovADijkstraCrsMPI(const InType &in) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BaranovADijkstraCrsMPI::ValidationImpl() {
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

bool BaranovADijkstraCrsMPI::PreProcessingImpl() {
  return true;
}

template <typename T>
std::vector<T> BaranovADijkstraCrsMPI::DijkstraParallelTemplate(int vertices, const std::vector<int> &row_ptr,
                                                                const std::vector<int> &col_idx,
                                                                const std::vector<T> &values, int source) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const T INF = std::numeric_limits<T>::max();
  std::vector<T> dist(vertices, INF);
  if (rank == 0) {
    std::vector<bool> visited(vertices, false);
    dist[source] = static_cast<T>(0);

    for (int i = 0; i < vertices; ++i) {
      T min_dist = INF;
      int u = -1;

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

      int start_idx = row_ptr[u];
      int end_idx = row_ptr[u + 1];

      for (int j = start_idx; j < end_idx; ++j) {
        int v = col_idx[j];
        T weight = values[j];

        if (!visited[v] && dist[u] + weight < dist[v]) {
          dist[v] = dist[u] + weight;
        }
      }
    }
  }

  MPI_Datatype mpi_type;
  if constexpr (std::is_same_v<T, int>) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same_v<T, float>) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    mpi_type = MPI_DOUBLE;
  } else {
    mpi_type = MPI_BYTE;
  }

  MPI_Bcast(dist.data(), vertices, mpi_type, 0, MPI_COMM_WORLD);

  return dist;
}

bool BaranovADijkstraCrsMPI::RunImpl() {
  try {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
      return false;
    }

    const auto &graph = std::get<GraphCRS>(GetInput());
    std::vector<double> weights_double;
    if (std::holds_alternative<int>(graph.weights)) {
      int weight = std::get<int>(graph.weights);
      weights_double.resize(graph.col_idx.size(), static_cast<double>(weight));
    } else if (std::holds_alternative<float>(graph.weights)) {
      float weight = std::get<float>(graph.weights);
      weights_double.resize(graph.col_idx.size(), static_cast<double>(weight));
    } else if (std::holds_alternative<double>(graph.weights)) {
      double weight = std::get<double>(graph.weights);
      weights_double.resize(graph.col_idx.size(), weight);
    } else {
      return false;
    }

    if (graph.row_ptr.size() != static_cast<size_t>(graph.vertices) + 1) {
      return false;
    }

    if (graph.col_idx.size() != weights_double.size()) {
      return false;
    }

    auto result =
        DijkstraParallelTemplate<double>(graph.vertices, graph.row_ptr, graph.col_idx, weights_double, graph.source);

    if (result.size() != static_cast<size_t>(graph.vertices)) {
      return false;
    }

    GetOutput() = result;
    return true;

  } catch (const std::exception &) {
    return false;
  } catch (...) {
    return false;
  }
}

bool BaranovADijkstraCrsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baranov_a_dijkstra_crs
