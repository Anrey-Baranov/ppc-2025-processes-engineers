#include "baranov_a_sign_alternations/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "baranov_a_sign_alternations/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baranov_a_sign_alternations {

BaranovASignAlternationsMPI::BaranovASignAlternationsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BaranovASignAlternationsMPI::ValidationImpl() {
  return !GetInput().empty() && (GetOutput() == 0);
}

bool BaranovASignAlternationsMPI::PreProcessingImpl() {
  return true;
}

bool BaranovASignAlternationsMPI::RunImpl() {
  const auto &input = GetInput();
  int total_alternations = 0;

  int world_size, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (input.size() < 2) {
    if (world_rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  int pairs_count = input.size() - 1;
  int pairs_per_process = pairs_count / world_size;
  int remainder = pairs_count % world_size;
  int start_pair, end_pair;

  if (world_rank < remainder) {
    start_pair = world_rank * (pairs_per_process + 1);
    end_pair = start_pair + pairs_per_process + 1;
  } else {
    start_pair = remainder * (pairs_per_process + 1) + (world_rank - remainder) * pairs_per_process;
    end_pair = start_pair + pairs_per_process;
  }

  int locale_alternations = 0;
  for (int i = start_pair; i < end_pair && i < pairs_count; i++) {
    int current = input[i];
    int next = input[i + 1];

    if (current != 0 && next != 0) {
      if ((current > 0 && next < 0) || (current < 0 && next > 0)) {
        locale_alternations++;
      }
    }
  }

  if (world_rank == 0) {
    total_alternations = locale_alternations;

    for (int i = 1; i < world_size; i++) {
      int received_alternations;
      MPI_Recv(&received_alternations, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      total_alternations += received_alternations;
    }

    GetOutput() = total_alternations;

  } else {
    MPI_Send(&locale_alternations, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

  

bool BaranovASignAlternationsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baranov_a_sign_alternations
