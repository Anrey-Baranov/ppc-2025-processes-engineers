#include "baranov_a_dijkstra_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"

namespace baranov_a_dijkstra_crs {

namespace {
void CalculateVertexDistribution(int world_rank, int world_size, int total_vertices, int &local_start, int &local_end,
                                 int &local_count) {
  int vertices_per_process = total_vertices / world_size;
  int remainder = total_vertices % world_size;

  if (world_rank < remainder) {
    local_start = world_rank * (vertices_per_process + 1);
    local_end = local_start + vertices_per_process + 1;
  } else {
    local_start = remainder * (vertices_per_process + 1) + (world_rank - remainder) * vertices_per_process;
    local_end = local_start + vertices_per_process;
  }
  local_count = local_end - local_start;
}
}  // namespace

BaranovADijkstraCRSMPI::BaranovADijkstraCRSMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool BaranovADijkstraCRSMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.num_vertices <= 0) {
    return false;
  }
  if (input.source_vertex < 0 || input.source_vertex >= input.num_vertices) {
    return false;
  }
  if (input.offsets.size() != static_cast<size_t>(input.num_vertices + 1)) {
    return false;
  }
  return true;
}

bool BaranovADijkstraCRSMPI::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  return true;
}

void BaranovADijkstraCRSMPI::DistributeGraphData() {
  const auto &graph = GetInput();
  const int total_vertices = graph.num_vertices;

  int local_start, local_end;
  CalculateVertexDistribution(world_rank, world_size, total_vertices, local_start, local_end, local_num_vertices);

  if (local_end > total_vertices) {
    local_end = total_vertices;
    local_num_vertices = local_end - local_start;
  }

  if (local_num_vertices <= 0 || local_start >= total_vertices) {
    local_offsets.clear();
    local_columns.clear();
    local_values.clear();
    vertex_ownership.resize(total_vertices);
    for (int i = 0; i < total_vertices; ++i) {
      vertex_ownership[i] = i % world_size;
    }
    return;
  }
  vertex_ownership.resize(total_vertices);
  for (int i = 0; i < total_vertices; ++i) {
    vertex_ownership[i] = i % world_size;
  }

  local_offsets.resize(local_num_vertices + 1);

  for (int i = 0; i <= local_num_vertices; ++i) {
    int global_idx = local_start + i;
    if (global_idx <= total_vertices) {
      local_offsets[i] = graph.offsets[global_idx] - graph.offsets[local_start];
    } else {
      local_offsets[i] = graph.offsets[total_vertices] - graph.offsets[local_start];
    }
  }

  int start_edge = graph.offsets[local_start];
  int end_edge = (local_end <= total_vertices) ? graph.offsets[local_end] : graph.offsets[total_vertices];
  int total_edges = end_edge - start_edge;

  if (total_edges > 0) {
    local_columns.resize(total_edges);
    local_values.resize(total_edges);

    for (int i = 0; i < total_edges; ++i) {
      if (start_edge + i < static_cast<int>(graph.columns.size())) {
        local_columns[i] = graph.columns[start_edge + i];
        local_values[i] = graph.values[start_edge + i];
      }
    }
  } else {
    local_columns.clear();
    local_values.clear();
  }
}

bool BaranovADijkstraCRSMPI::RunImpl() {
  const auto &graph = GetInput();
  const int total_vertices = graph.num_vertices;
  const int source = graph.source_vertex;

  DistributeGraphData();
  int local_start, local_end;
  CalculateVertexDistribution(world_rank, world_size, total_vertices, local_start, local_end, local_num_vertices);

  if (local_end > total_vertices) {
    local_end = total_vertices;
    local_num_vertices = local_end - local_start;
  }

  std::vector<double> global_dist(total_vertices, std::numeric_limits<double>::infinity());
  bool i_own_source = false;
  if (local_num_vertices > 0 && source >= local_start && source < local_end) {
    i_own_source = true;
    global_dist[source] = 0.0;
  }

  int source_owner = -1;
  int my_has_source = i_own_source ? world_rank : -1;
  MPI_Allreduce(&my_has_source, &source_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (source_owner >= 0) {
    MPI_Bcast(global_dist.data(), total_vertices, MPI_DOUBLE, source_owner, MPI_COMM_WORLD);
  }

  if (global_dist.empty()) {
    GetOutput() = std::vector<double>();
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  std::vector<double> local_dist = global_dist;

  for (int iter = 0; iter < total_vertices; ++iter) {
    bool changed = false;

    if (local_num_vertices > 0 && local_start < total_vertices) {
      for (int i = 0; i < local_num_vertices; ++i) {
        int global_v = local_start + i;
        if (global_v >= total_vertices) {
          continue;
        }
        if (local_dist[global_v] < std::numeric_limits<double>::infinity()) {
          if (i < static_cast<int>(local_offsets.size()) - 1) {
            int start = local_offsets[i];
            int end = local_offsets[i + 1];

            for (int idx = start; idx < end; ++idx) {
              if (idx < static_cast<int>(local_columns.size())) {
                int neighbor = local_columns[idx];
                double weight = local_values[idx];
                if (neighbor >= 0 && neighbor < total_vertices) {
                  double new_dist = local_dist[global_v] + weight;
                  if (new_dist < local_dist[neighbor]) {
                    local_dist[neighbor] = new_dist;
                    changed = true;
                  }
                }
              }
            }
          }
        }
      }
    }
    int global_changed = 0;
    int local_changed = changed ? 1 : 0;
    MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (global_changed == 0) {
      break;
    }
    MPI_Allreduce(local_dist.data(), global_dist.data(), total_vertices, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (!global_dist.empty() && !local_dist.empty() && global_dist.size() == local_dist.size()) {
      local_dist = global_dist;
    }
  }
  GetOutput() = global_dist;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool BaranovADijkstraCRSMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baranov_a_dijkstra_crs
