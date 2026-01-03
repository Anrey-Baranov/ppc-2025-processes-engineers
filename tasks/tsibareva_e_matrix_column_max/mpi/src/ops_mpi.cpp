#include "tsibareva_e_matrix_column_max/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"

namespace tsibareva_e_matrix_column_max {

TsibarevaEMatrixColumnMaxMPI::TsibarevaEMatrixColumnMaxMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = std::vector<std::vector<int>>(in);
  GetOutput() = std::vector<int>();
}

bool TsibarevaEMatrixColumnMaxMPI::ValidationImpl() {
  return true;
}

bool TsibarevaEMatrixColumnMaxMPI::PreProcessingImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = std::vector<int>();
    final_result_ = std::vector<int>();
    return true;
  }

  size_t first_row_size = matrix[0].size();
  for (size_t i = 1; i < matrix.size(); ++i) {
    if (matrix[i].size() != first_row_size) {
      GetOutput() = std::vector<int>();
      final_result_ = std::vector<int>();
      return true;
    }
  }

  final_result_ = std::vector<int>(GetInput()[0].size(), 0);
  GetOutput() = std::vector<int>(GetInput()[0].size(), 0);
  return true;
}

bool TsibarevaEMatrixColumnMaxMPI::RunImpl() {
  if (GetOutput().empty()) {
    return true;
  }

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &matrix = GetInput();
  size_t rows_count = matrix.size();
  size_t cols_count = matrix[0].size();

  std::vector<int> local_maxs;

  for (auto col = static_cast<size_t>(world_rank); col < cols_count; col += static_cast<size_t>(world_size)) {
    int max_val = matrix[0][col];
    for (size_t row = 1; row < rows_count; ++row) {
      max_val = std::max(matrix[row][col], max_val);
    }
    local_maxs.push_back(max_val);
  }

  if (world_rank == 0) {
    CollectResultsFromAllProcesses(local_maxs, world_size, cols_count);
  } else {
    if (!local_maxs.empty()) {
      MPI_Send(local_maxs.data(), static_cast<int>(local_maxs.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }

  return true;
}

void TsibarevaEMatrixColumnMaxMPI::CollectResultsFromAllProcesses(const std::vector<int> &local_maxs, int world_size,
                                                                  size_t cols_count) {
  final_result_.resize(cols_count);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  size_t idx = 0;
  for (size_t col = 0; col < cols_count && idx < local_maxs.size(); col += world_size) {
    final_result_[col] = local_maxs[idx++];
  }

  for (int proc = 1; proc < world_size; proc++) {
    int proc_pass = 0;

    for (size_t col = proc; col < cols_count; col += world_size) {
      proc_pass++;
    }

    if (proc_pass <= 0) {
      continue;
    }

    std::vector<int> proc_maxs(static_cast<size_t>(proc_pass));
    MPI_Recv(proc_maxs.data(), proc_pass, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    size_t proc_idx = 0;
    for (size_t col = proc; col < cols_count; col += world_size) {
      final_result_[col] = proc_maxs[proc_idx++];
    }
  }
}

bool TsibarevaEMatrixColumnMaxMPI::PostProcessingImpl() {
  if (GetOutput().empty()) {
    return true;
  }

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &matrix = GetInput();
  size_t cols_count = matrix[0].size();

  if (world_rank == 0) {
    GetOutput() = final_result_;
  } else {
    GetOutput().resize(cols_count);
  }

  MPI_Bcast(GetOutput().data(), static_cast<int>(cols_count), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

}  // namespace tsibareva_e_matrix_column_max
