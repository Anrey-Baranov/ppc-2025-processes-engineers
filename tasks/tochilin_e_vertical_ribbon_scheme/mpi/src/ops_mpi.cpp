#include "tochilin_e_vertical_ribbon_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "tochilin_e_vertical_ribbon_scheme/common/include/common.hpp"

namespace tochilin_e_vertical_ribbon_scheme {

TochilinEVerticalRibbonSchemeMPI::TochilinEVerticalRibbonSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TochilinEVerticalRibbonSchemeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &input = GetInput();
    if (input.rows <= 0 || input.cols <= 0) {
      return false;
    }
    if (std::cmp_not_equal(input.matrix.size(), static_cast<std::size_t>(input.rows) * input.cols)) {
      return false;
    }
    if (std::cmp_not_equal(input.vector.size(), input.cols)) {
      return false;
    }
  }
  return true;
}

bool TochilinEVerticalRibbonSchemeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &input = GetInput();
    rows_ = input.rows;
    cols_ = input.cols;
    matrix_ = input.matrix;
    vector_ = input.vector;
  }
  return true;
}

bool TochilinEVerticalRibbonSchemeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base_cols = cols_ / size;
  int extra_cols = cols_ % size;

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  std::vector<int> vec_counts(size);
  std::vector<int> vec_displs(size);

  int offset = 0;
  int vec_offset = 0;
  for (int i = 0; i < size; i++) {
    int local_cols = base_cols + (i < extra_cols ? 1 : 0);
    send_counts[i] = local_cols * rows_;
    displs[i] = offset;
    vec_counts[i] = local_cols;
    vec_displs[i] = vec_offset;
    offset += send_counts[i];
    vec_offset += local_cols;
  }

  int local_cols = base_cols + (rank < extra_cols ? 1 : 0);
  std::vector<double> local_matrix(static_cast<std::size_t>(local_cols) * rows_);
  std::vector<double> local_vector(local_cols);

  MPI_Scatterv(matrix_.data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_matrix.data(), send_counts[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(vector_.data(), vec_counts.data(), vec_displs.data(), MPI_DOUBLE, local_vector.data(), vec_counts[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_result(rows_, 0.0);
  for (int j = 0; j < local_cols; j++) {
    for (int i = 0; i < rows_; i++) {
      local_result[i] += local_matrix[static_cast<std::size_t>(j * rows_) + i] * local_vector[j];
    }
  }

  result_.resize(rows_);
  MPI_Reduce(local_result.data(), result_.data(), rows_, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(result_.data(), rows_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool TochilinEVerticalRibbonSchemeMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace tochilin_e_vertical_ribbon_scheme
