#include "kopilov_d_ring_2/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <utility>
#include <vector>

#include "kopilov_d_ring_2/common/include/common.hpp"

namespace kopilov_d_ring_2 {

KopilovDRingMPI::KopilovDRingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KopilovDRingMPI::ValidationImpl() {
  return true;
}

bool KopilovDRingMPI::PreProcessingImpl() {
  GetOutput().data.clear();
  return true;
}

bool KopilovDRingMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<int> current_data;
  if (rank == 0) {
    current_data = GetInput().data;
  }

  auto apply_offset = [rank](std::vector<int> &data) {
    for (int &val : data) {
      val += rank;
    }
  };

  if (world_size > 1) {
    const int next_proc = (rank + 1) % world_size;
    const int prev_proc = (rank - 1 + world_size) % world_size;

    if (rank == 0) {
      apply_offset(current_data);
      auto size = static_cast<int>(current_data.size());
      MPI_Send(&size, 1, MPI_INT, next_proc, 0, MPI_COMM_WORLD);
      MPI_Send(current_data.data(), size, MPI_INT, next_proc, 1, MPI_COMM_WORLD);
      MPI_Recv(&size, 1, MPI_INT, prev_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      current_data.resize(size);
      MPI_Recv(current_data.data(), size, MPI_INT, prev_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      int size = 0;
      MPI_Recv(&size, 1, MPI_INT, prev_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      current_data.resize(size);
      MPI_Recv(current_data.data(), size, MPI_INT, prev_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      apply_offset(current_data);

      MPI_Send(&size, 1, MPI_INT, next_proc, 0, MPI_COMM_WORLD);
      MPI_Send(current_data.data(), size, MPI_INT, next_proc, 1, MPI_COMM_WORLD);
    }
  } else {
    apply_offset(current_data);
  }

  int final_size = 0;
  if (rank == 0) {
    final_size = static_cast<int>(current_data.size());
  }
  MPI_Bcast(&final_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    current_data.resize(final_size);
  }

  MPI_Bcast(current_data.data(), final_size, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput().data = std::move(current_data);

  return true;
}

bool KopilovDRingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kopilov_d_ring_2
