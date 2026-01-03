#include "likhanov_m_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>

#include "likhanov_m_elem_vec_sum/common/include/common.hpp"

namespace likhanov_m_elem_vec_sum {

LikhanovMElemVecSumMPI::LikhanovMElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LikhanovMElemVecSumMPI::ValidationImpl() {
  return GetInput() >= 0;
}

bool LikhanovMElemVecSumMPI::PreProcessingImpl() {
  return true;
}

bool LikhanovMElemVecSumMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int64_t n = GetInput();

  int64_t base = n / size;
  int64_t rem = n % size;

  int64_t local_begin = (rank * base) + std::min<int64_t>(rank, rem) + 1;

  int64_t local_end = local_begin + base - 1 + (rank < rem ? 1 : 0);

  int64_t local_sum = 0;
  for (int64_t i = local_begin; i <= local_end; ++i) {
    local_sum += i;
  }

  int64_t global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_sum;
  }

  return true;
}

bool LikhanovMElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace likhanov_m_elem_vec_sum
