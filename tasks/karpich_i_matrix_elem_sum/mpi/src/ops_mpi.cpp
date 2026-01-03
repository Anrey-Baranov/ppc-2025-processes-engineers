#include "karpich_i_matrix_elem_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "karpich_i_matrix_elem_sum/common/include/common.hpp"

namespace karpich_i_matrix_elem_sum {

KarpichIMatrixElemSumMPI::KarpichIMatrixElemSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KarpichIMatrixElemSumMPI::ValidationImpl() {
  std::size_t n = std::get<0>(GetInput());
  std::size_t m = std::get<1>(GetInput());
  std::vector<int> val = std::get<2>(GetInput());

  return (n > 0) && (m > 0) && (val.size() == (n * m));
}

bool KarpichIMatrixElemSumMPI::PreProcessingImpl() {
  return true;
}

bool KarpichIMatrixElemSumMPI::RunImpl() {
  std::vector<int> &val = std::get<2>(GetInput());

  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int total_elements = 0;
  if (rank == 0) {
    total_elements = static_cast<int>(val.size());
  }
  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int elements_per_proc = total_elements / mpi_size;
  int remainder = total_elements % mpi_size;

  std::vector<int> send_counts(mpi_size, elements_per_proc);
  std::vector<int> displacements(mpi_size, 0);

  for (int i = 0; i < remainder; ++i) {
    send_counts[i]++;
  }

  displacements[0] = 0;
  for (int i = 1; i < mpi_size; ++i) {
    displacements[i] = displacements[i - 1] + send_counts[i - 1];
  }

  int local_size = send_counts[rank];
  std::vector<int> local_data(local_size);

  MPI_Scatterv(val.data(), send_counts.data(), displacements.data(), MPI_INT, local_data.data(), local_size, MPI_INT, 0,
               MPI_COMM_WORLD);

  std::int64_t local_sum = std::accumulate(local_data.begin(), local_data.end(), static_cast<std::int64_t>(0));

  std::int64_t global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_sum, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

  GetOutput() = global_sum;
  return true;
}

bool KarpichIMatrixElemSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace karpich_i_matrix_elem_sum
