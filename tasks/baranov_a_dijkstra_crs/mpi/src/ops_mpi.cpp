#include "baranov_a_dijkstra_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <variant>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"

namespace baranov_a_dijkstra_crs {

BaranovADijkstraCrsMPI::BaranovADijkstraCrsMPI(const InType &in) {
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
    if (graph.row_ptr[0] != 0) {
      return false;
    }

    for (size_t i = 1; i < graph.row_ptr.size(); ++i) {
      if (graph.row_ptr[i] < graph.row_ptr[i - 1]) {
        return false;
      }
      if (graph.row_ptr[i] > static_cast<int>(graph.col_idx.size())) {
        return false;
      }
    }

    for (int col : graph.col_idx) {
      if (col < 0 || col >= graph.vertices) {
        return false;
      }
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
void BaranovADijkstraCrsMPI::TreeBroadcast(std::vector<T> &data, int count, int root, MPI_Datatype datatype,
                                           MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (count == 0) {
    return;
  }
  if (rank == root) {
    for (int i = 0; i < size; i++) {
      if (i != root) {
        MPI_Send(data.data(), count, datatype, i, 0, comm);
      }
    }
  } else {
    MPI_Recv(data.data(), count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
  }
}

template <typename T>
void BaranovADijkstraCrsMPI::TreeAllReduceMin(std::vector<T> &local_data, std::vector<T> &global_data, int count,
                                              MPI_Datatype datatype, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (count == 0) {
    return;
  }
  if (rank == 0) {
    std::memcpy(global_data.data(), local_data.data(), count * sizeof(T));

    for (int i = 1; i < size; i++) {
      std::vector<T> temp_buf(count);
      MPI_Recv(temp_buf.data(), count, datatype, i, 0, comm, MPI_STATUS_IGNORE);
      for (int j = 0; j < count; j++) {
        if (temp_buf[j] < global_data[j]) {
          global_data[j] = temp_buf[j];
        }
      }
    }
    for (int i = 1; i < size; i++) {
      MPI_Send(global_data.data(), count, datatype, i, 1, comm);
    }
  } else {
    MPI_Send(local_data.data(), count, datatype, 0, 0, comm);
    MPI_Recv(global_data.data(), count, datatype, 0, 1, comm, MPI_STATUS_IGNORE);
  }
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
  TreeBroadcast(dist, vertices, 0, mpi_type, MPI_COMM_WORLD);

  std::vector<bool> visited(vertices, false);

  for (int i = 0; i < vertices; ++i) {
    T local_min = INF;
    int local_u = -1;

    for (int v = 0; v < vertices; ++v) {
      if (!visited[v] && dist[v] < local_min) {
        local_min = dist[v];
        local_u = v;
      }
    }
    T global_min = INF;
    int global_u = -1;
    MPI_Allreduce(&local_min, &global_min, 1, mpi_type, MPI_MIN, MPI_COMM_WORLD);
    if (global_min == INF) {
      break;
    }
    int candidate_vertex = -1;

    if (local_u != -1) {
      if constexpr (std::is_integral_v<T>) {
        if (local_min == global_min) {
          candidate_vertex = local_u;
        }
      } else {
        const T epsilon = std::numeric_limits<T>::epsilon() * 10;
        if (std::abs(local_min - global_min) <= epsilon * std::max(std::abs(local_min), std::abs(global_min))) {
          candidate_vertex = local_u;
        }
      }
    }
    MPI_Allreduce(&candidate_vertex, &global_u, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (global_u == -1) {
      break;
    }

    visited[global_u] = true;
    int start = row_ptr[global_u];
    int end = row_ptr[global_u + 1];

    for (int j = start; j < end; ++j) {
      int v = col_idx[j];
      T weight = values[j];

      if (!visited[v]) {
        T new_dist = (dist[global_u] == INF) ? INF : dist[global_u] + weight;
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
        }
      }
    }
    std::vector<T> global_dist(vertices);
    TreeAllReduceMin(dist, global_dist, vertices, mpi_type, MPI_COMM_WORLD);
    dist.swap(global_dist);
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
  return true;
}

}  // namespace baranov_a_dijkstra_crs
