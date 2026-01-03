#include "tabalaev_a_cannon_mat_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <vector>

#include "tabalaev_a_cannon_mat_mul/common/include/common.hpp"

namespace tabalaev_a_cannon_mat_mul {

TabalaevACannonMatMulMPI::TabalaevACannonMatMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TabalaevACannonMatMulMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int validation = 0;

  if (world_rank == 0) {
    const size_t n = std::get<0>(GetInput());
    const auto &a = std::get<1>(GetInput());
    const auto &b = std::get<2>(GetInput());
    if (n > 0 && a.size() == n * n && b.size() == n * n) {
      validation = 1;
    }
  }

  MPI_Bcast(&validation, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return validation != 0;
}

bool TabalaevACannonMatMulMPI::PreProcessingImpl() {
  return true;
}

bool TabalaevACannonMatMulMPI::RunImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int n = 0;
  if (world_rank == 0) {
    n = static_cast<int>(std::get<0>(GetInput()));
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int q = static_cast<int>(std::sqrt(world_size));

  if (q * q != world_size || n % q != 0) {
    const size_t result_size = static_cast<size_t>(n) * static_cast<size_t>(n);
    std::vector<double> result(result_size, 0.0);
    if (world_rank == 0) {
      auto &a = std::get<1>(GetInput());
      auto &b = std::get<2>(GetInput());
      LocalMatrixMultiply(a, b, result, n);
    }
    MPI_Bcast(result.data(), static_cast<int>(result_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    GetOutput() = result;

    return true;
  }

  int block_size = n / q;
  int block_elems = block_size * block_size;

  std::array<int, 2> dims = {q, q};
  std::array<int, 2> periods = {1, 1};
  MPI_Comm grid_comm = MPI_COMM_NULL;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 1, &grid_comm);

  int grid_rank = 0;
  MPI_Comm_rank(grid_comm, &grid_rank);

  std::array<int, 2> coords = {0, 0};
  MPI_Cart_coords(grid_comm, grid_rank, 2, coords.data());

  int row = coords[0];
  int col = coords[1];

  MPI_Datatype block_type = MPI_DATATYPE_NULL;
  MPI_Type_vector(block_size, block_size, n, MPI_DOUBLE, &block_type);
  MPI_Type_commit(&block_type);
  MPI_Datatype resized_block = MPI_DATATYPE_NULL;
  MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block);
  MPI_Type_commit(&resized_block);

  std::vector<double> local_a(block_elems);
  std::vector<double> local_b(block_elems);
  std::vector<double> local_c(block_elems, 0.0);

  std::vector<int> counts(world_size, 1);
  std::vector<int> displs(world_size);

  if (grid_rank == 0) {
    for (int i = 0; i < q; ++i) {
      for (int j = 0; j < q; ++j) {
        displs[(i * q) + j] = (i * n * block_size) + (j * block_size);
      }
    }
    auto &a = std::get<1>(GetInput());
    auto &b = std::get<2>(GetInput());
    MPI_Scatterv(a.data(), counts.data(), displs.data(), resized_block, local_a.data(), block_elems, MPI_DOUBLE, 0,
                 grid_comm);
    MPI_Scatterv(b.data(), counts.data(), displs.data(), resized_block, local_b.data(), block_elems, MPI_DOUBLE, 0,
                 grid_comm);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, resized_block, local_a.data(), block_elems, MPI_DOUBLE, 0, grid_comm);
    MPI_Scatterv(nullptr, nullptr, nullptr, resized_block, local_b.data(), block_elems, MPI_DOUBLE, 0, grid_comm);
  }

  int left = 0;
  int right = 0;
  int up = 0;
  int down = 0;

  MPI_Cart_shift(grid_comm, 1, 1, &left, &right);
  MPI_Cart_shift(grid_comm, 0, 1, &up, &down);

  for (int i = 0; i < row; ++i) {
    MPI_Sendrecv_replace(local_a.data(), block_elems, MPI_DOUBLE, left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
  }
  for (int i = 0; i < col; ++i) {
    MPI_Sendrecv_replace(local_b.data(), block_elems, MPI_DOUBLE, up, 1, down, 1, grid_comm, MPI_STATUS_IGNORE);
  }

  for (int k = 0; k < q; ++k) {
    LocalMatrixMultiply(local_a, local_b, local_c, block_size);
    MPI_Sendrecv_replace(local_a.data(), block_elems, MPI_DOUBLE, left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(local_b.data(), block_elems, MPI_DOUBLE, up, 1, down, 1, grid_comm, MPI_STATUS_IGNORE);
  }

  std::vector<double> global_result(static_cast<size_t>(n) * static_cast<size_t>(n));
  MPI_Gatherv(local_c.data(), block_elems, MPI_DOUBLE, global_result.data(), counts.data(), displs.data(),
              resized_block, 0, grid_comm);

  MPI_Bcast(global_result.data(), n * n, MPI_DOUBLE, 0, grid_comm);
  GetOutput() = global_result;

  MPI_Type_free(&resized_block);
  MPI_Type_free(&block_type);
  MPI_Comm_free(&grid_comm);
  return true;
}

bool TabalaevACannonMatMulMPI::PostProcessingImpl() {
  return true;
}

void TabalaevACannonMatMulMPI::LocalMatrixMultiply(const std::vector<double> &a, const std::vector<double> &b,
                                                   std::vector<double> &c, int n) {
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      double temp = a[(i * n) + k];
      for (int j = 0; j < n; ++j) {
        c[(i * n) + j] += temp * b[(k * n) + j];
      }
    }
  }
}

}  // namespace tabalaev_a_cannon_mat_mul
