#include "samoylenko_i_conj_grad_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "samoylenko_i_conj_grad_method/common/include/common.hpp"

namespace samoylenko_i_conj_grad_method {

SamoylenkoIConjGradMethodMPI::SamoylenkoIConjGradMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SamoylenkoIConjGradMethodMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int valid = 0;
  if (world_rank == 0) {
    valid = (GetInput().first > 0 && (GetInput().second >= 0 && GetInput().second <= 2) && GetOutput().empty()) ? 1 : 0;
  }

  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid == 1;
}

bool SamoylenkoIConjGradMethodMPI::PreProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    GetOutput().clear();
  }

  return true;
}

namespace {

void CalculateDistribution(size_t size, int world_size, std::vector<int> &row_counts, std::vector<int> &row_displs) {
  int proc_rows = static_cast<int>(size) / world_size;
  int extra_rows = static_cast<int>(size) % world_size;
  int disp = 0;

  for (int proc = 0; proc < world_size; ++proc) {
    row_counts[proc] = proc_rows + (proc < extra_rows ? 1 : 0);
    row_displs[proc] = disp;
    disp += row_counts[proc];
  }
}

std::vector<double> BuildLocalMatrix(size_t size, int local_rows, int local_start, int variant) {
  std::vector<double> local_matrix(local_rows * size, 0.0);

  for (int i = 0; i < local_rows; ++i) {
    size_t global_row = local_start + i;
    switch (variant) {
      case 0: {
        local_matrix[(static_cast<size_t>(i) * size) + global_row] = 4.0;
        if (global_row > 0) {
          local_matrix[(static_cast<size_t>(i) * size) + (global_row - 1)] = 1.0;
        }
        if (global_row + 1 < size) {
          local_matrix[(static_cast<size_t>(i) * size) + (global_row + 1)] = 1.0;
        }
        break;
      }

      case 1: {
        local_matrix[(static_cast<size_t>(i) * size) + global_row] = 5.0;
        break;
      }

      case 2: {
        local_matrix[(static_cast<size_t>(i) * size) + global_row] = 3.0;
        size_t anti_col = size - 1 - global_row;
        if (anti_col != global_row) {
          local_matrix[(static_cast<size_t>(i) * size) + anti_col] = -1.0;
        }
        break;
      }

      default:
        break;
    }
  }
  return local_matrix;
}

void LocalMatrixVectorMult(size_t size, int local_rows, const std::vector<double> &local_matrix,
                           const std::vector<double> &vector, std::vector<double> &local_result) {
  for (int i = 0; i < local_rows; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < size; ++j) {
      sum += local_matrix[(static_cast<size_t>(i) * size) + j] * vector[j];
    }
    local_result[i] = sum;
  }
}

double LocalDotProduct(int local_rows, const std::vector<double> &local_first,
                       const std::vector<double> &local_second) {
  double sum = 0.0;
  for (int i = 0; i < local_rows; ++i) {
    sum += local_first[i] * local_second[i];
  }
  return sum;
}

void ConjugateGradient(size_t size, int local_rows, const std::vector<double> &local_matrix,
                       const std::vector<double> &local_vector, const std::vector<int> &row_counts,
                       const std::vector<int> &row_displs, std::vector<double> &local_x) {
  const double eps = 1e-7;
  const int iters = 2000;

  std::vector<double> local_res(local_rows);
  std::vector<double> local_dir(local_rows);
  std::vector<double> local_matdir(local_rows);

  std::vector<double> x(size);
  std::vector<double> dir(size);

  MPI_Allgatherv(local_x.data(), local_rows, MPI_DOUBLE, x.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);
  LocalMatrixVectorMult(size, local_rows, local_matrix, x, local_matdir);

  for (int i = 0; i < local_rows; ++i) {
    local_res[i] = local_vector[i] - local_matdir[i];
    local_dir[i] = local_res[i];
  }

  double local_res_dot = LocalDotProduct(local_rows, local_res, local_res);
  double res_dot = 0.0;
  MPI_Allreduce(&local_res_dot, &res_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  for (int it = 0; it < iters; ++it) {
    if (std::sqrt(res_dot) < eps) {
      break;
    }

    MPI_Allgatherv(local_dir.data(), local_rows, MPI_DOUBLE, dir.data(), row_counts.data(), row_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    LocalMatrixVectorMult(size, local_rows, local_matrix, dir, local_matdir);

    double local_dir_dot = LocalDotProduct(local_rows, local_dir, local_matdir);
    double dir_dot = 0.0;
    MPI_Allreduce(&local_dir_dot, &dir_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (std::fabs(dir_dot) < 1e-15) {
      break;  // so we dont divide by 0
    }

    double step = res_dot / dir_dot;

    for (int i = 0; i < local_rows; ++i) {
      local_x[i] += step * local_dir[i];
      local_res[i] -= step * local_matdir[i];
    }

    double local_res_dot_new = LocalDotProduct(local_rows, local_res, local_res);
    double res_dot_new = 0.0;
    MPI_Allreduce(&local_res_dot_new, &res_dot_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double conj_coef = res_dot_new / res_dot;

    for (int i = 0; i < local_rows; ++i) {
      local_dir[i] = local_res[i] + (conj_coef * local_dir[i]);
    }

    res_dot = res_dot_new;
  }
}

}  // namespace

bool SamoylenkoIConjGradMethodMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int n = 0;
  int variant = 0;
  if (world_rank == 0) {
    n = GetInput().first;
    variant = GetInput().second;
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&variant, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 0) {
    return false;
  }

  auto size = static_cast<size_t>(n);

  std::vector<int> row_counts(world_size);
  std::vector<int> row_displs(world_size);
  CalculateDistribution(size, world_size, row_counts, row_displs);

  int local_rows = row_counts[world_rank];
  int local_start = row_displs[world_rank];

  std::vector<double> local_matrix = BuildLocalMatrix(size, local_rows, local_start, variant);
  std::vector<double> vector(size);
  if (world_rank == 0) {
    for (size_t i = 0; i < size; ++i) {
      vector[i] = 1.0;
    }
  }

  std::vector<double> local_vector(local_rows);
  MPI_Scatterv(vector.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE, local_vector.data(), local_rows,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_x(local_rows, 0.0);

  ConjugateGradient(size, local_rows, local_matrix, local_vector, row_counts, row_displs, local_x);

  std::vector<double> x(size);
  MPI_Gatherv(local_x.data(), local_rows, MPI_DOUBLE, x.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = x;
  }

  return true;
}

bool SamoylenkoIConjGradMethodMPI::PostProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    return !GetOutput().empty();
  }

  return true;
}

}  // namespace samoylenko_i_conj_grad_method
