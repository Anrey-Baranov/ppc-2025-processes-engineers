#include "shilin_n_gauss_band_horizontal_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "shilin_n_gauss_band_horizontal_scheme/common/include/common.hpp"

namespace shilin_n_gauss_band_horizontal_scheme {

ShilinNGaussBandHorizontalSchemeMPI::ShilinNGaussBandHorizontalSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    InType &input_ref = GetInput();
    input_ref.clear();
    input_ref.reserve(in.size());
    for (const auto &row : in) {
      input_ref.push_back(row);
    }
  }
  GetOutput() = std::vector<double>();
}

bool ShilinNGaussBandHorizontalSchemeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int validation_result = 1;

  if (rank == 0) {
    const InType &input = GetInput();
    validation_result = ValidateInput(input);
  }

  MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return validation_result != 0;
}

int ShilinNGaussBandHorizontalSchemeMPI::ValidateInput(const InType &input) {
  if (input.empty()) {
    return 0;
  }

  size_t n = input.size();
  if (n == 0) {
    return 0;
  }

  size_t cols = input[0].size();
  if (cols < n + 1) {
    return 0;
  }

  for (size_t i = 1; i < n; ++i) {
    if (input[i].size() != cols) {
      return 0;
    }
  }

  return 1;
}

bool ShilinNGaussBandHorizontalSchemeMPI::PreProcessingImpl() {
  GetOutput() = std::vector<double>();
  return true;
}

bool ShilinNGaussBandHorizontalSchemeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  InType augmented_matrix;
  size_t n = 0;
  size_t cols = 0;

  if (rank == 0) {
    augmented_matrix = GetInput();
    n = augmented_matrix.size();
    cols = (n > 0) ? augmented_matrix[0].size() : 0;
  }

  int n_int = static_cast<int>(n);
  int cols_int = static_cast<int>(cols);
  MPI_Bcast(&n_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  n = static_cast<size_t>(n_int);
  cols = static_cast<size_t>(cols_int);

  if (n == 0 || cols < n + 1) {
    return false;
  }

  InType local_matrix;
  std::vector<int> global_to_local(n, -1);
  DistributeRows(augmented_matrix, n, cols, rank, size, local_matrix, global_to_local);

  if (!ForwardEliminationMPI(local_matrix, global_to_local, n, cols, rank, size)) {
    return false;
  }

  std::vector<double> x = BackSubstitutionMPI(local_matrix, global_to_local, n, cols, rank, size);

  GetOutput() = x;

  return true;
}

void ShilinNGaussBandHorizontalSchemeMPI::DistributeRows(const InType &augmented_matrix, size_t n, size_t cols,
                                                         int rank, int size, InType &local_matrix,
                                                         std::vector<int> &global_to_local) {
  // распределение строк по round-robin: строка i -> процесс i % size
  int local_rows = 0;
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(i) % size == rank) {
      local_rows++;
    }
  }

  local_matrix = InType(local_rows, std::vector<double>(cols));
  // маппинг глобальных индексов строк на локальные индексы
  int local_idx = 0;
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(i) % size == rank) {
      global_to_local[i] = local_idx;
      if (rank == 0) {
        local_matrix[local_idx] = augmented_matrix[i];
      } else {
        MPI_Recv(local_matrix[local_idx].data(), static_cast<int>(cols), MPI_DOUBLE, 0, static_cast<int>(i),
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      local_idx++;
    } else if (rank == 0) {
      MPI_Send(augmented_matrix[i].data(), static_cast<int>(cols), MPI_DOUBLE, static_cast<int>(i) % size,
               static_cast<int>(i), MPI_COMM_WORLD);
    }
  }
}

bool ShilinNGaussBandHorizontalSchemeMPI::ForwardEliminationMPI(InType &local_matrix,
                                                                const std::vector<int> &global_to_local, size_t n,
                                                                size_t cols, int rank, int size) {
  for (size_t k = 0; k < n; ++k) {
    // определение процесса-владельца ведущей строки
    int owner_process = static_cast<int>(k) % size;

    std::vector<double> pivot_row(cols);
    if (rank == owner_process) {
      int local_k = global_to_local[k];
      if (local_k >= 0 && static_cast<size_t>(local_k) < local_matrix.size()) {
        pivot_row = local_matrix[static_cast<size_t>(local_k)];
      }
    }

    // рассылка ведущей строки всем процессам для исключения
    MPI_Bcast(pivot_row.data(), static_cast<int>(cols), MPI_DOUBLE, owner_process, MPI_COMM_WORLD);

    if (std::abs(pivot_row[k]) < 1e-10) {
      return false;
    }

    EliminateColumnMPI(local_matrix, global_to_local, k, n, cols, pivot_row);
  }
  return true;
}

void ShilinNGaussBandHorizontalSchemeMPI::EliminateColumnMPI(InType &local_matrix,
                                                             const std::vector<int> &global_to_local, size_t k,
                                                             size_t n, size_t cols,
                                                             const std::vector<double> &pivot_row) {
  // исключение элементов в локальных строках
  for (size_t i = 0; i < local_matrix.size(); ++i) {
    size_t global_i = GetGlobalIndex(global_to_local, i, n);

    if (global_i > k && std::abs(local_matrix[i][k]) > 1e-10) {
      double factor = local_matrix[i][k] / pivot_row[k];
      for (size_t j = k; j < cols; ++j) {
        local_matrix[i][j] -= factor * pivot_row[j];
      }
    }
  }
}

size_t ShilinNGaussBandHorizontalSchemeMPI::GetGlobalIndex(const std::vector<int> &global_to_local, size_t local_idx,
                                                           size_t n) {
  // восстановление глобального индекса из локального
  const int local_idx_int = static_cast<int>(local_idx);
  for (size_t gi = 0; gi < n; ++gi) {
    if (global_to_local[gi] >= 0 && global_to_local[gi] == local_idx_int) {
      return gi;
    }
  }
  return 0;
}

std::vector<double> ShilinNGaussBandHorizontalSchemeMPI::BackSubstitutionMPI(const InType &local_matrix,
                                                                             const std::vector<int> &global_to_local,
                                                                             size_t n, size_t cols, int rank,
                                                                             int size) {
  // обратный ход с синхронизацией между процессами
  std::vector<double> x(n, 0.0);

  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    double sum = 0.0;
    int owner_process = i % size;

    if (rank == owner_process) {
      int local_i = global_to_local[static_cast<size_t>(i)];
      if (local_i >= 0 && static_cast<size_t>(local_i) < local_matrix.size()) {
        for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
          sum += local_matrix[static_cast<size_t>(local_i)][j] * x[j];
        }
        x[static_cast<size_t>(i)] = (local_matrix[static_cast<size_t>(local_i)][cols - 1] - sum) /
                                    local_matrix[static_cast<size_t>(local_i)][static_cast<size_t>(i)];
      }
    }

    // рассылка вычисленного значения для использования в следующих итерациях
    MPI_Bcast(&x[static_cast<size_t>(i)], 1, MPI_DOUBLE, owner_process, MPI_COMM_WORLD);
  }

  return x;
}

bool ShilinNGaussBandHorizontalSchemeMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return (rank == 0) ? !GetOutput().empty() : true;
}

}  // namespace shilin_n_gauss_band_horizontal_scheme
