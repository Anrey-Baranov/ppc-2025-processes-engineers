#include "vasiliev_m_bellman_ford_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <limits>
#include <vector>

#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"

namespace vasiliev_m_bellman_ford_crs {

VasilievMBellmanFordCrsMPI::VasilievMBellmanFordCrsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMBellmanFordCrsMPI::ValidationImpl() {
  const auto &in = GetInput();

  if (in.row_ptr.empty() || in.col_ind.empty() || in.vals.empty()) {
    return false;
  }

  if (in.col_ind.size() != in.vals.size()) {
    return false;
  }

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;
  if (vertices <= 0) {
    return false;
  }

  if (in.source < 0 || in.source >= vertices) {
    return false;
  }

  return true;
}

bool VasilievMBellmanFordCrsMPI::PreProcessingImpl() {
  const auto &in = GetInput();
  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;

  const int inf = std::numeric_limits<int>::max();
  GetOutput().assign(vertices, inf);
  GetOutput()[in.source] = 0;

  return true;
}

bool VasilievMBellmanFordCrsMPI::RunImpl() {
  const auto &in = GetInput();
  auto &dist = GetOutput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    CalcCountsAndDispls(vertices, size, counts, displs);
  }

  MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  int start = displs[rank];
  int end = start + counts[rank];

  std::vector<int> local_dist(vertices);

  for (int i = 0; i < vertices - 1; i++) {
    local_dist = dist;

    int local_updated = RelaxLocalEdges(start, end, in, dist, local_dist);

    MPI_Allreduce(local_dist.data(), dist.data(), vertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    int global_updated = 0;
    MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_updated == 0) {
      break;
    }
  }

  return true;
}

bool VasilievMBellmanFordCrsMPI::PostProcessingImpl() {
  return true;
}

void VasilievMBellmanFordCrsMPI::CalcCountsAndDispls(int n, int size, std::vector<int> &counts,
                                                     std::vector<int> &displs) {
  int chunk = n / size;
  int remain = n % size;

  for (int i = 0; i < size; i++) {
    counts[i] = chunk + (i < remain ? 1 : 0);
  }

  displs[0] = 0;
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
}

int VasilievMBellmanFordCrsMPI::RelaxLocalEdges(int start, int end, const InType &in, const std::vector<int> &dist,
                                                std::vector<int> &local_dist) {
  const int inf = std::numeric_limits<int>::max();
  int local_updated = 0;

  for (int vertex = start; vertex < end; vertex++) {
    if (dist[vertex] == inf) {
      continue;
    }

    for (int edge = in.row_ptr[vertex]; edge < in.row_ptr[vertex + 1]; edge++) {
      int v = in.col_ind[edge];
      int w = in.vals[edge];

      if (local_dist[v] > dist[vertex] + w) {
        local_dist[v] = dist[vertex] + w;
        local_updated = 1;
      }
    }
  }

  return local_updated;
}

}  // namespace vasiliev_m_bellman_ford_crs
