#include "baranov_a_dijkstra_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <limits>
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
    local_start = (remainder * (vertices_per_process + 1)) + ((world_rank - remainder) * vertices_per_process);
    local_end = local_start + vertices_per_process;
  }
  local_count = local_end - local_start;
}

void InitializeVertexOwnership(std::vector<int> &vertex_ownership, int total_vertices, int world_size) {
  vertex_ownership.resize(total_vertices);
  for (int i = 0; i < total_vertices; ++i) {
    vertex_ownership[i] = i % world_size;
  }
}

void CopyLocalEdges(const GraphData &graph, int local_start, int local_end, int total_vertices,
                    std::vector<int> &local_columns, std::vector<double> &local_values) {
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

void InitializeGlobalDist(std::vector<double> &global_dist, int total_vertices, int source, bool i_own_source) {
  if (total_vertices > 0) {
    global_dist.resize(total_vertices, std::numeric_limits<double>::infinity());
    if (i_own_source && !global_dist.empty() && source >= 0 && static_cast<std::size_t>(source) < global_dist.size()) {
      global_dist[source] = 0.0;
    }
  }
}

bool ProcessNeighbors(const std::vector<double> &local_dist, int global_v, const std::vector<int> &local_offsets, int i,
                      const std::vector<int> &local_columns, const std::vector<double> &local_values,
                      int total_vertices, std::vector<double> &new_dist) {
  bool changed = false;
  int start = local_offsets[i];
  int end = local_offsets[i + 1];

  for (int idx = start; idx < end; ++idx) {
    if (static_cast<std::size_t>(idx) >= local_columns.size()) {
      continue;
    }

    int neighbor = local_columns[idx];
    double weight = local_values[idx];

    if (neighbor < 0 || neighbor >= total_vertices) {
      continue;
    }

    double new_distance = local_dist[global_v] + weight;
    if (new_distance < new_dist[neighbor]) {
      new_dist[neighbor] = new_distance;
      changed = true;
    }
  }
  return changed;
}

bool ProcessLocalVertices(const std::vector<double> &local_dist, const std::vector<int> &local_offsets,
                          const std::vector<int> &local_columns, const std::vector<double> &local_values,
                          int local_start, int local_num_vertices, int total_vertices, std::vector<double> &new_dist) {
  bool changed = false;

  if (local_num_vertices <= 0 || local_start >= total_vertices || local_dist.empty()) {
    return changed;
  }

  for (int i = 0; i < local_num_vertices; ++i) {
    int global_v = local_start + i;
    if (global_v >= total_vertices) {
      continue;
    }

    if (local_dist[global_v] >= std::numeric_limits<double>::infinity()) {
      continue;
    }

    if (i >= static_cast<int>(local_offsets.size()) - 1) {
      continue;
    }

    changed = ProcessNeighbors(local_dist, global_v, local_offsets, i, local_columns, local_values, total_vertices,
                               new_dist) ||
              changed;
  }

  return changed;
}

bool UpdateDistances(std::vector<double> &global_dist, std::vector<double> &local_dist, std::vector<double> &new_dist,
                     int total_vertices) {
  MPI_Allreduce(new_dist.data(), global_dist.data(), total_vertices, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  if (!global_dist.empty() && !local_dist.empty() && global_dist.size() == local_dist.size()) {
    local_dist.assign(global_dist.begin(), global_dist.end());
  }

  if (!global_dist.empty() && !new_dist.empty() && global_dist.size() == new_dist.size()) {
    new_dist.assign(global_dist.begin(), global_dist.end());
  }

  return true;
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
  if (input.offsets.size() != static_cast<std::size_t>(input.num_vertices) + 1) {
    return false;
  }
  return true;
}

bool BaranovADijkstraCRSMPI::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  return true;
}

void BaranovADijkstraCRSMPI::DistributeGraphData() {
  const auto &graph = GetInput();
  const int total_vertices = graph.num_vertices;

  int local_start = 0;
  int local_end = 0;
  CalculateVertexDistribution(world_rank_, world_size_, total_vertices, local_start, local_end, local_num_vertices_);

  if (local_end > total_vertices) {
    local_end = total_vertices;
    local_num_vertices_ = local_end - local_start;
  }

  if (local_num_vertices_ <= 0 || local_start >= total_vertices) {
    local_offsets_.clear();
    local_columns_.clear();
    local_values_.clear();
    InitializeVertexOwnership(vertex_ownership_, total_vertices, world_size_);
    return;
  }

  InitializeVertexOwnership(vertex_ownership_, total_vertices, world_size_);
  local_offsets_.resize(static_cast<std::size_t>(local_num_vertices_) + 1);

  for (int i = 0; i <= local_num_vertices_; ++i) {
    int global_idx = local_start + i;
    if (global_idx <= total_vertices) {
      local_offsets_[static_cast<std::size_t>(i)] =
          graph.offsets[static_cast<std::size_t>(global_idx)] - graph.offsets[static_cast<std::size_t>(local_start)];
    } else {
      local_offsets_[static_cast<std::size_t>(i)] = graph.offsets[static_cast<std::size_t>(total_vertices)] -
                                                    graph.offsets[static_cast<std::size_t>(local_start)];
    }
  }

  CopyLocalEdges(graph, local_start, local_end, total_vertices, local_columns_, local_values_);
}

bool BaranovADijkstraCRSMPI::RunImpl() {
  const auto &graph = GetInput();
  const int total_vertices = graph.num_vertices;
  const int source = graph.source_vertex;

  if (total_vertices <= 0) {
    GetOutput() = std::vector<double>();
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  DistributeGraphData();

  int local_start = 0;
  int local_end = 0;
  CalculateVertexDistribution(world_rank_, world_size_, total_vertices, local_start, local_end, local_num_vertices_);

  if (local_end > total_vertices) {
    local_end = total_vertices;
    local_num_vertices_ = local_end - local_start;
  }

  std::vector<double> global_dist;
  bool i_own_source = (total_vertices > 0 && local_num_vertices_ > 0 && source >= local_start && source < local_end);
  InitializeGlobalDist(global_dist, total_vertices, source, i_own_source);

  int source_owner = -1;
  int my_has_source = i_own_source ? world_rank_ : -1;
  MPI_Allreduce(&my_has_source, &source_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (source_owner >= 0 && !global_dist.empty()) {
    MPI_Bcast(global_dist.data(), total_vertices, MPI_DOUBLE, source_owner, MPI_COMM_WORLD);
  }

  if (global_dist.empty()) {
    GetOutput() = std::vector<double>();
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  std::vector<double> local_dist = global_dist;
  std::vector<double> new_dist = global_dist;

  for (int iter = 0; iter < total_vertices; ++iter) {
    bool changed = ProcessLocalVertices(local_dist, local_offsets_, local_columns_, local_values_, local_start,
                                        local_num_vertices_, total_vertices, new_dist);

    int global_changed = 0;
    int local_changed_int = changed ? 1 : 0;
    MPI_Allreduce(&local_changed_int, &global_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (global_changed == 0) {
      break;
    }

    if (!new_dist.empty() && !global_dist.empty()) {
      UpdateDistances(global_dist, local_dist, new_dist, total_vertices);
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
