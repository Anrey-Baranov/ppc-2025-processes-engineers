#include "ermakov_a_ring/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "ermakov_a_ring/common/include/common.hpp"

namespace ermakov_a_ring {

ErmakovATestTaskMPI::ErmakovATestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ErmakovATestTaskMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size > 0;
}

bool ErmakovATestTaskMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ErmakovATestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  std::vector<int> payload = input.data;
  const int src = ((input.source % size) + size) % size;
  const int dst = ((input.dest % size) + size) % size;

  const int cw_dist = (dst - src + size) % size;
  const int cc_dist = (src - dst + size) % size;

  bool clockwise = (cw_dist <= cc_dist);
  int steps = cc_dist;
  int next = (rank - 1 + size) % size;
  int prev = (rank + 1) % size;
  int dist_from_src = (src - rank + size) % size;

  if (clockwise) {
    steps = cw_dist;
    next = (rank + 1) % size;
    prev = (rank - 1 + size) % size;
    dist_from_src = (rank - src + size) % size;
  }

  std::vector<int> path;
  if (rank == src) {
    path.push_back(src);
    if (src != dst) {
      const int path_sz = static_cast<int>(path.size());
      const int data_sz = static_cast<int>(payload.size());
      MPI_Send(payload.data(), data_sz, MPI_INT, next, 100, MPI_COMM_WORLD);
      MPI_Send(&path_sz, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
      MPI_Send(path.data(), path_sz, MPI_INT, next, 1, MPI_COMM_WORLD);
    }
  }

  bool is_in_path = (dist_from_src > 0 && dist_from_src <= steps);
  if (is_in_path) {
    int path_sz = 0;
    const int data_sz = static_cast<int>(payload.size());
    MPI_Recv(payload.data(), data_sz, MPI_INT, prev, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&path_sz, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    path.resize(static_cast<size_t>(path_sz));
    MPI_Recv(path.data(), path_sz, MPI_INT, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    path.push_back(rank);

    if (rank != dst) {
      const int next_path_sz = static_cast<int>(path.size());
      MPI_Send(payload.data(), data_sz, MPI_INT, next, 100, MPI_COMM_WORLD);
      MPI_Send(&next_path_sz, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
      MPI_Send(path.data(), next_path_sz, MPI_INT, next, 1, MPI_COMM_WORLD);
    }
  }

  int final_path_sz = static_cast<int>(path.size());
  MPI_Bcast(&final_path_sz, 1, MPI_INT, dst, MPI_COMM_WORLD);
  if (rank != dst) {
    path.resize(static_cast<size_t>(final_path_sz));
  }
  MPI_Bcast(path.data(), final_path_sz, MPI_INT, dst, MPI_COMM_WORLD);

  GetOutput() = path;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ErmakovATestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ermakov_a_ring
