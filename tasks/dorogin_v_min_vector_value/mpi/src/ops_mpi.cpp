#include "dorogin_v_min_vector_value/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "dorogin_v_min_vector_value/common/include/common.hpp"

namespace dorogin_v_min_vector_value {

DoroginVMinVectorValueMPI::DoroginVMinVectorValueMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DoroginVMinVectorValueMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool DoroginVMinVectorValueMPI::PreProcessingImpl() {
  return true;
}

bool DoroginVMinVectorValueMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &data = GetInput();
  const int data_size = static_cast<int>(data.size());

  // Calculate chunk size for each process
  const int chunk_size = data_size / size;
  const int remainder = data_size % size;
  const int local_size = chunk_size + (rank < remainder ? 1 : 0);

  // Distribute data: rank 0 sends chunks to other ranks
  std::vector<int> local_data(local_size);
  if (rank == 0) {
    // Rank 0 keeps its chunk
    for (int i = 0; i < local_size; ++i) {
      local_data[i] = data[i];
    }
    // Send chunks to other ranks
    int offset = local_size;
    for (int dest = 1; dest < size; ++dest) {
      const int dest_size = chunk_size + (dest < remainder ? 1 : 0);
      MPI_Send(data.data() + offset, dest_size, MPI_INT, dest, 0, MPI_COMM_WORLD);
      offset += dest_size;
    }
  } else {
    // Other ranks receive their chunk
    MPI_Recv(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Find local minimum
  int local_min = 0;
  if (local_size > 0) {
    const auto it_min = std::ranges::min_element(local_data);
    local_min = (it_min != local_data.end()) ? *it_min : 0;
  } else {
    local_min = std::numeric_limits<int>::max();
  }

  // Reduce to find global minimum
  int global_min = 0;
  MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  // Broadcast result to all ranks
  MPI_Bcast(&global_min, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_min;
  return true;
}

bool DoroginVMinVectorValueMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dorogin_v_min_vector_value
