#include "samoylenko_i_simple_iter_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "samoylenko_i_simple_iter_method/common/include/common.hpp"

namespace samoylenko_i_simple_iter_method {

SamoylenkoISimpleIterMethodMPI::SamoylenkoISimpleIterMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SamoylenkoISimpleIterMethodMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int valid = 0;
  if (world_rank == 0) {
    valid = (GetInput() > 0 && GetOutput().empty()) ? 1 : 0;
  }

  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid == 1;
}

bool SamoylenkoISimpleIterMethodMPI::PreProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    GetOutput().clear();
  }

  return true;
}

namespace {

void CalculateDistribution(int n, int world_size, std::vector<int> &row_counts, std::vector<int> &row_displs) {
  int proc_rows = n / world_size;
  int extra_rows = n % world_size;
  int disp = 0;

  for (int proc = 0; proc < world_size; ++proc) {
    row_counts[proc] = proc_rows + (proc < extra_rows ? 1 : 0);
    row_displs[proc] = disp;
    disp += row_counts[proc];
  }
}

std::vector<double> BuildLocalMatrix(int n, int local_rows, int local_start) {
  auto size = static_cast<size_t>(n);
  std::vector<double> local_matrix(static_cast<size_t>(local_rows) * size, 0.0);

  for (int i = 0; i < local_rows; ++i) {
    int global_row = local_start + i;
    local_matrix[(static_cast<size_t>(i) * size) + global_row] = 4.0;

    if (global_row > 0) {
      local_matrix[(static_cast<size_t>(i) * size) + (global_row - 1)] = 1.0;
    }
    if (global_row + 1 < n) {
      local_matrix[(static_cast<size_t>(i) * size) + (global_row + 1)] = 1.0;
    }
  }

  return local_matrix;
}

void PerformIterations(int n, int local_rows, int local_start, const std::vector<double> &local_matrix,
                       const std::vector<double> &local_vector, const std::vector<int> &row_counts,
                       const std::vector<int> &row_displs, std::vector<double> &x) {
  auto size = static_cast<size_t>(n);
  const double tau = 0.2;
  const double eps = 1e-7;
  const int iters = 2000;

  std::vector<double> x_old(size);
  std::vector<double> local_x_new(local_rows);

  for (int it = 0; it < iters; ++it) {
    x_old = x;
    for (int i = 0; i < local_rows; ++i) {
      double ax_i = 0.0;
      for (size_t j = 0; j < size; ++j) {
        ax_i += local_matrix[(static_cast<size_t>(i) * size) + j] * x[j];
      }

      local_x_new[i] = x[local_start + i] - (tau * (ax_i - local_vector[i]));
    }

    MPI_Allgatherv(local_x_new.data(), local_rows, MPI_DOUBLE, x.data(), row_counts.data(), row_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double local_max = 0.0;
    for (int i = 0; i < local_rows; ++i) {
      local_max = std::max(local_max, std::fabs(x[local_start + i] - x_old[local_start + i]));
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (global_max < eps) {
      break;
    }
  }
}

}  // namespace

bool SamoylenkoISimpleIterMethodMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int n = 0;
  if (world_rank == 0) {
    n = GetInput();
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 0) {
    return false;
  }

  std::vector<int> row_counts(world_size);
  std::vector<int> row_displs(world_size);
  CalculateDistribution(n, world_size, row_counts, row_displs);

  int local_rows = row_counts[world_rank];
  int local_start = row_displs[world_rank];

  std::vector<double> local_matrix = BuildLocalMatrix(n, local_rows, local_start);
  std::vector<double> local_vector(local_rows, 1.0);
  std::vector<double> x(static_cast<size_t>(n), 0.0);

  PerformIterations(n, local_rows, local_start, local_matrix, local_vector, row_counts, row_displs, x);

  if (world_rank == 0) {
    GetOutput() = x;
  }

  return true;
}

bool SamoylenkoISimpleIterMethodMPI::PostProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    return !GetOutput().empty();
  }

  return true;
}

}  // namespace samoylenko_i_simple_iter_method
