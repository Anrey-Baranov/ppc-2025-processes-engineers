#include "baranov_a_sign_alternations/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
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

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (input.size() < 2) {
    GetOutput() = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int pairs_count = input.size() - 1;

  if (pairs_count < world_size) {
    int alternations_count = 0;

    if (world_rank == 0) {
      for (size_t i = 0; i < input.size() - 1; i++) {
        int current = input[i];
        int next = input[i + 1];
        if (current != 0 && next != 0) {
          if ((current > 0 && next < 0) || (current < 0 && next > 0)) {
            alternations_count++;
          }
        }
      }
    }

    MPI_Bcast(&alternations_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = alternations_count;
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

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
  int local_alternations = 0;
  for (int i = start_pair; i < end_pair && i < pairs_count; i++) {
    int current = input[i];
    int next = input[i + 1];

    if (current != 0 && next != 0) {
      if ((current > 0 && next < 0) || (current < 0 && next > 0)) {
        local_alternations++;
      }
    }
  }
  if (world_rank == 0) {
    int total_alternations = local_alternations;

    for (int i = 1; i < world_size; i++) {
      int received_alternations;
      MPI_Recv(&received_alternations, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      total_alternations += received_alternations;
    }

    GetOutput() = total_alternations;
    for (int i = 1; i < world_size; i++) {
      MPI_Send(&total_alternations, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Send(&local_alternations, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    int final_result;
    MPI_Recv(&final_result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    GetOutput() = final_result;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool BaranovASignAlternationsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baranov_a_sign_alternations
