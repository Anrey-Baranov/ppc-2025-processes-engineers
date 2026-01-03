#include "makoveeva_s_simple_iteration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "makoveeva_s_simple_iteration/common/include/common.hpp"

namespace makoveeva_s_simple_iteration {

namespace {
constexpr double kRelaxationFactor = 0.5;
constexpr double kEpsilon = 1e-6;
constexpr int kMaxIterations = 1000;

void CalculateRowsDistribution(int n, int size, std::vector<int> &row_counts, std::vector<int> &row_displs,
                               std::vector<int> &matrix_counts, std::vector<int> &matrix_displs) {
  int row_offset = 0;
  int matrix_offset = 0;
  for (int proc = 0; proc < size; ++proc) {
    int base_rows = n / size;
    int extra = (proc < (n % size)) ? 1 : 0;
    int proc_rows = base_rows + extra;

    row_counts[proc] = proc_rows;
    row_displs[proc] = row_offset;
    matrix_counts[proc] = proc_rows * n;
    matrix_displs[proc] = matrix_offset;

    row_offset += proc_rows;
    matrix_offset += proc_rows * n;
  }
}

void InitializeMatrixAndVector(std::vector<double> &flat_matrix, std::vector<double> &b, int n) {
  const auto matrix_size = static_cast<size_t>(n) * static_cast<size_t>(n);
  flat_matrix.resize(matrix_size, 0.0);
  b.resize(static_cast<size_t>(n), 0.0);

  for (int i = 0; i < n; ++i) {
    const auto i_idx = static_cast<size_t>(i);
    const auto n_idx = static_cast<size_t>(n);

    flat_matrix[(i_idx * n_idx) + i_idx] = static_cast<double>(n) + 5.0;

    for (int j = 0; j < n; ++j) {
      if (i != j) {
        const auto j_idx = static_cast<size_t>(j);
        flat_matrix[(i_idx * n_idx) + j_idx] = 1.0 / (static_cast<double>(std::abs(i - j)) + 1.0);
      }
    }

    for (int j = 0; j < n; ++j) {
      const auto j_idx = static_cast<size_t>(j);
      b[i_idx] += flat_matrix[(i_idx * n_idx) + j_idx] * static_cast<double>(j + 1);
    }
  }
}

double ComputeLocalProduct(const std::vector<double> &local_matrix, const std::vector<double> &x,
                           const std::vector<double> &local_b, std::vector<double> &local_x_new, int local_rows,
                           int start_row, int n) {
  double local_diff = 0.0;

  for (int i = 0; i < local_rows; ++i) {
    const auto i_idx = static_cast<size_t>(i);
    const auto n_idx = static_cast<size_t>(n);

    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      const auto j_idx = static_cast<size_t>(j);
      sum += local_matrix[(i_idx * n_idx) + j_idx] * x[j_idx];
    }

    const int global_i = start_row + i;
    const auto global_idx = static_cast<size_t>(global_i);

    local_x_new[i_idx] =
        x[global_idx] + (kRelaxationFactor * (local_b[i_idx] - sum) / local_matrix[(i_idx * n_idx) + global_idx]);

    double diff = local_x_new[i_idx] - x[global_idx];
    local_diff += diff * diff;
  }

  return local_diff;
}

int ComputeFinalResult(const std::vector<double> &x, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += x[static_cast<size_t>(i)];
  }
  return static_cast<int>(std::round(sum));
}

}  // namespace

MakoveevaSSimpleIterationMPI::MakoveevaSSimpleIterationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MakoveevaSSimpleIterationMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n = GetInput();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    GetInput() = n;
  }

  int is_valid = 0;
  if (rank == 0) {
    is_valid = (GetInput() > 0) ? 1 : 0;
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid != 0;
}

bool MakoveevaSSimpleIterationMPI::PreProcessingImpl() {
  return true;
}

bool MakoveevaSSimpleIterationMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = GetInput();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    GetInput() = n;
  }
  if (n <= 0) {
    return false;
  }

  std::vector<int> row_counts(static_cast<size_t>(size));
  std::vector<int> row_displs(static_cast<size_t>(size));
  std::vector<int> matrix_counts(static_cast<size_t>(size));
  std::vector<int> matrix_displs(static_cast<size_t>(size));

  CalculateRowsDistribution(n, size, row_counts, row_displs, matrix_counts, matrix_displs);

  const int local_rows = row_counts[rank];
  const int start_row = row_displs[rank];

  std::vector<double> flat_matrix;
  std::vector<double> b;

  if (rank == 0) {
    InitializeMatrixAndVector(flat_matrix, b, n);
  }

  std::vector<double> local_matrix(static_cast<size_t>(local_rows) * static_cast<size_t>(n));
  MPI_Scatterv(rank == 0 ? flat_matrix.data() : nullptr, matrix_counts.data(), matrix_displs.data(), MPI_DOUBLE,
               local_matrix.data(), local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_b(static_cast<size_t>(local_rows));
  MPI_Scatterv(rank == 0 ? b.data() : nullptr, row_counts.data(), row_displs.data(), MPI_DOUBLE, local_b.data(),
               local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(static_cast<size_t>(n), 0.0);
  std::vector<double> x_new(static_cast<size_t>(n), 0.0);
  std::vector<double> local_x_new(static_cast<size_t>(local_rows), 0.0);

  bool converged = false;

  for (int iteration = 0; iteration < kMaxIterations && !converged; ++iteration) {
    const double local_diff = ComputeLocalProduct(local_matrix, x, local_b, local_x_new, local_rows, start_row, n);

    MPI_Allgatherv(local_x_new.data(), local_rows, MPI_DOUBLE, x_new.data(), row_counts.data(), row_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double global_diff_sq = 0.0;
    MPI_Allreduce(&local_diff, &global_diff_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const double global_diff = std::sqrt(global_diff_sq);
    converged = (global_diff < kEpsilon);

    x.swap(x_new);
  }

  if (rank == 0) {
    GetOutput() = ComputeFinalResult(x, n);
  }

  MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool MakoveevaSSimpleIterationMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace makoveeva_s_simple_iteration
