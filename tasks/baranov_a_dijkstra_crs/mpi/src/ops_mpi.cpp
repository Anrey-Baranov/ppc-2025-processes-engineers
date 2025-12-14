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

  int type_size;
  MPI_Type_size(datatype, &type_size);

  if (rank == 0) {
    std::memcpy(global_data.data(), local_data.data(), static_cast<size_t>(count) * type_size);

    for (int i = 1; i < size; i++) {
      std::vector<unsigned char> temp_buf(static_cast<size_t>(count) * type_size);
      MPI_Recv(temp_buf.data(), count, datatype, i, 0, comm, MPI_STATUS_IGNORE);

      T *temp = reinterpret_cast<T *>(temp_buf.data());
      T *global = global_data.data();

      for (int j = 0; j < count; j++) {
        if (temp[j] < global[j]) {
          global[j] = temp[j];
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

  // Каждый процесс вычисляет свою порцию вершин
  int local_vertices = vertices / size;
  int remainder = vertices % size;

  int start_idx = rank * local_vertices + std::min(rank, remainder);
  int end_idx = start_idx + local_vertices + (rank < remainder ? 1 : 0);

  // Распределение начальных расстояний
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

  // Broadcast начального массива расстояний
  TreeBroadcast(dist, vertices, 0, mpi_type, MPI_COMM_WORLD);

  std::vector<bool> visited(vertices, false);

  for (int i = 0; i < vertices; ++i) {
    // Находим минимальную непосещенную вершину локально
    T local_min = INF;
    int local_u = -1;

    for (int v = start_idx; v < end_idx; ++v) {
      if (!visited[v] && dist[v] < local_min) {
        local_min = dist[v];
        local_u = v;
      }
    }

    // Находим глобальный минимум среди всех процессов
    struct {
      T dist;
      int vertex;
    } local_data, global_data;

    local_data.dist = local_min;
    local_data.vertex = local_u;

    MPI_Datatype minloc_type;
    if constexpr (std::is_same_v<T, int>) {
      minloc_type = MPI_2INT;
    } else if constexpr (std::is_same_v<T, float>) {
      minloc_type = MPI_FLOAT_INT;
    } else {
      minloc_type = MPI_DOUBLE_INT;
    }

    MPI_Allreduce(&local_data, &global_data, 1, minloc_type, MPI_MINLOC, MPI_COMM_WORLD);

    int global_u = global_data.vertex;
    T global_min_dist = global_data.dist;

    if (global_u == -1 || global_min_dist == INF) {
      break;
    }

    visited[global_u] = true;

    // Обновляем расстояния для соседей выбранной вершины
    int start = row_ptr[global_u];
    int end = row_ptr[global_u + 1];

    for (int j = start; j < end; ++j) {
      int v = col_idx[j];
      T weight = values[j];

      if (!visited[v] && dist[global_u] + weight < dist[v]) {
        dist[v] = dist[global_u] + weight;
      }
    }

    // Синхронизируем обновленные расстояния между процессами
    std::vector<T> global_dist(vertices);
    TreeAllReduceMin(dist, global_dist, vertices, mpi_type, MPI_COMM_WORLD);
    dist = global_dist;
  }

  return dist;
}

bool BaranovADijkstraCrsMPI::RunImpl() {
  try {
    const auto &graph = std::get<GraphCRS>(GetInput());

    // Преобразуем веса в правильный тип
    if (std::holds_alternative<int>(graph.weights)) {
      int weight = std::get<int>(graph.weights);
      std::vector<int> values(graph.col_idx.size(), weight);
      auto result = DijkstraParallelTemplate<int>(graph.vertices, graph.row_ptr, graph.col_idx, values, graph.source);
      GetOutput() = result;
    } else if (std::holds_alternative<float>(graph.weights)) {
      float weight = std::get<float>(graph.weights);
      std::vector<float> values(graph.col_idx.size(), weight);
      auto result = DijkstraParallelTemplate<float>(graph.vertices, graph.row_ptr, graph.col_idx, values, graph.source);
      GetOutput() = result;
    } else if (std::holds_alternative<double>(graph.weights)) {
      double weight = std::get<double>(graph.weights);
      std::vector<double> values(graph.col_idx.size(), weight);
      auto result =
          DijkstraParallelTemplate<double>(graph.vertices, graph.row_ptr, graph.col_idx, values, graph.source);
      GetOutput() = result;
    } else {
      return false;
    }

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

// Явная инстанциация шаблонных функций
template void BaranovADijkstraCrsMPI::TreeBroadcast<int>(std::vector<int> &, int, int, MPI_Datatype, MPI_Comm);
template void BaranovADijkstraCrsMPI::TreeBroadcast<float>(std::vector<float> &, int, int, MPI_Datatype, MPI_Comm);
template void BaranovADijkstraCrsMPI::TreeBroadcast<double>(std::vector<double> &, int, int, MPI_Datatype, MPI_Comm);

template void BaranovADijkstraCrsMPI::TreeAllReduceMin<int>(std::vector<int> &, std::vector<int> &, int, MPI_Datatype,
                                                            MPI_Comm);
template void BaranovADijkstraCrsMPI::TreeAllReduceMin<float>(std::vector<float> &, std::vector<float> &, int,
                                                              MPI_Datatype, MPI_Comm);
template void BaranovADijkstraCrsMPI::TreeAllReduceMin<double>(std::vector<double> &, std::vector<double> &, int,
                                                               MPI_Datatype, MPI_Comm);

}  // namespace baranov_a_dijkstra_crs
