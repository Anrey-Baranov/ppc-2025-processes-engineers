#include "baranov_a_dijkstra_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <variant>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baranov_a_dijkstra_crs {

BaranovADijkstraCrsMPI::BaranovADijkstraCrsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

BaranovADijkstraCrsMPI::~BaranovADijkstraCrsMPI() noexcept {
  try {
    ppc::util::DestructorFailureFlag::Unset();
  } catch (...) {
  }
}

bool BaranovADijkstraCrsMPI::ValidationImpl() {
  try {
    if (!std::holds_alternative<GraphCRS>(GetInput())) {
      return false;
    }

    const auto &graph = std::get<GraphCRS>(GetInput());
    return ValidateGraph(graph);

  } catch (...) {
    return false;
  }
}

bool BaranovADijkstraCrsMPI::ValidateGraph(const GraphCRS &graph) {
  if (graph.vertices <= 0) {
    return false;
  }
  if (graph.source < 0 || static_cast<size_t>(graph.source) >= static_cast<size_t>(graph.vertices)) {
    return false;
  }
  if (graph.row_ptr.empty() || graph.row_ptr.size() != static_cast<size_t>(graph.vertices) + 1) {
    return false;
  }

  return true;
}

bool BaranovADijkstraCrsMPI::PreProcessingImpl() {
  return true;
}

template <typename T>
void BaranovADijkstraCrsMPI::TreeBroadcast(std::vector<T> &data, int root, MPI_Datatype mpi_type) {
  MPI_Bcast(data.data(), static_cast<int>(data.size()), mpi_type, root, MPI_COMM_WORLD);
}

template <typename T>
void BaranovADijkstraCrsMPI::TreeAllReduceMin(std::vector<T> &data, MPI_Datatype mpi_type) {
  MPI_Allreduce(MPI_IN_PLACE, data.data(), static_cast<int>(data.size()), mpi_type, MPI_MIN, MPI_COMM_WORLD);
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
    dist[source] = static_cast<T>(0);
  }

  MPI_Datatype mpi_type;
  if constexpr (std::is_same_v<T, int>) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same_v<T, float>) {
    mpi_type = MPI_FLOAT;
  } else {
    mpi_type = MPI_DOUBLE;
  }

  MPI_Bcast(dist.data(), static_cast<int>(dist.size()), mpi_type, 0, MPI_COMM_WORLD);

  std::vector<bool> visited(vertices, false);
  for (int i = 0; i < vertices; ++i) {
    int local_u = -1;
    T local_min = INF;

    for (int v = 0; v < vertices; ++v) {
      if (!visited[v] && dist[v] < local_min) {
        local_min = dist[v];
        local_u = v;
      }
    }
    struct {
      T dist;
      int vertex;
    } local, global;

    local.dist = local_min;
    local.vertex = local_u;
    MPI_Datatype minloc_type;
    if constexpr (std::is_same_v<T, int>) {
      minloc_type = MPI_2INT;
    } else if constexpr (std::is_same_v<T, float>) {
      minloc_type = MPI_FLOAT_INT;
    } else {
      minloc_type = MPI_DOUBLE_INT;
    }
    MPI_Allreduce(&local, &global, 1, minloc_type, MPI_MINLOC, MPI_COMM_WORLD);

    int global_u = global.vertex;
    T global_min_dist = global.dist;
    if (global_u == -1 || global_min_dist == INF) {
      break;
    }
    visited[global_u] = true;
    int start = row_ptr[global_u];
    int end = row_ptr[global_u + 1];

    for (int j = start; j < end; ++j) {
      int v = col_idx[j];
      T weight = values[j];

      if (!visited[v]) {
        T new_dist = dist[global_u] + weight;
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, dist.data(), static_cast<int>(dist.size()), mpi_type, MPI_MIN, MPI_COMM_WORLD);
  }

  return dist;
}

bool BaranovADijkstraCrsMPI::RunImpl() {
  try {
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

    auto result =
        DijkstraParallelTemplate<double>(graph.vertices, graph.row_ptr, graph.col_idx, weights_double, graph.source);

    GetOutput() = result;
    MPI_Barrier(MPI_COMM_WORLD);

    return true;

  } catch (const std::exception &e) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cerr << "Error in RunImpl: " << e.what() << std::endl;
    }
    return false;
  } catch (...) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cerr << "Unknown error in RunImpl" << std::endl;
    }
    return false;
  }
}

bool BaranovADijkstraCrsMPI::PostProcessingImpl() {
  try {
    ppc::util::DestructorFailureFlag::Unset();
  } catch (...) {
  }
  return true;
}

template void BaranovADijkstraCrsMPI::TreeBroadcast<double>(std::vector<double> &, int, MPI_Datatype);
template void BaranovADijkstraCrsMPI::TreeAllReduceMin<double>(std::vector<double> &, MPI_Datatype);

}  // namespace baranov_a_dijkstra_crs
