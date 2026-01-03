#include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/common/include/common.hpp"

namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector {

TsibarevaERibbonHorizontalMatrixMultVectorMPI::TsibarevaERibbonHorizontalMatrixMultVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  GetInput() = in;

  if (world_rank == 0) {
    input_matrix_ = std::get<0>(GetInput());
    rows_ = std::get<1>(GetInput());
    cols_ = std::get<2>(GetInput());
    local_vector_ = std::get<3>(GetInput());
  } else {
    input_matrix_ = std::vector<int>();
    rows_ = -1;
    cols_ = -1;
    local_vector_ = std::vector<int>();
  }
  GetOutput() = std::vector<int>();
}

bool TsibarevaERibbonHorizontalMatrixMultVectorMPI::ValidationImpl() {
  return true;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorMPI::PreProcessingImpl() {
  return true;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  BroadcastMatrixDimensions();

  if (rows_ == 0 || cols_ == 0) {
    GetOutput() = std::vector<int>();
    return true;
  }

  BroadcastVector();

  int rows_base = rows_ / world_size;
  int remainder = rows_ % world_size;
  local_rows_ = rows_base + (world_rank < remainder ? 1 : 0);

  std::vector<int> send_counts;
  std::vector<int> displacements;
  PrepareScatterParameters(world_rank, world_size, send_counts, displacements);

  ScatterMatrixData(world_rank, send_counts, displacements);

  std::vector<int> local_result = CalculateMultiplyLocalPart();

  std::vector<int> recv_counts(world_size);
  std::vector<int> displs(world_size);
  std::vector<int> global_result(rows_);

  int mdisplace = 0;
  for (int i = 0; i < world_size; i++) {
    int proc_rows = rows_base + (i < remainder ? 1 : 0);
    recv_counts[i] = proc_rows;
    displs[i] = mdisplace;
    mdisplace += proc_rows;
  }

  MPI_Allgatherv(local_result.data(), local_rows_, MPI_INT, global_result.data(), recv_counts.data(), displs.data(),
                 MPI_INT, MPI_COMM_WORLD);

  GetOutput() = global_result;
  return true;
}

void TsibarevaERibbonHorizontalMatrixMultVectorMPI::BroadcastMatrixDimensions() {
  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void TsibarevaERibbonHorizontalMatrixMultVectorMPI::BroadcastVector() {
  local_vector_.resize(cols_);
  MPI_Bcast(local_vector_.data(), cols_, MPI_INT, 0, MPI_COMM_WORLD);
}

void TsibarevaERibbonHorizontalMatrixMultVectorMPI::PrepareScatterParameters(int world_rank, int world_size,
                                                                             std::vector<int> &send_counts,
                                                                             std::vector<int> &displacements) const {
  send_counts.resize(world_size);
  displacements.resize(world_size);

  if (world_rank == 0) {
    int rows_base = rows_ / world_size;
    int remainder = rows_ % world_size;
    int displ = 0;

    for (int i = 0; i < world_size; i++) {
      int proc_rows = rows_base + (i < remainder ? 1 : 0);
      send_counts[i] = proc_rows * cols_;
      displacements[i] = displ;
      displ += send_counts[i];
    }
  }

  MPI_Bcast(send_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void TsibarevaERibbonHorizontalMatrixMultVectorMPI::ScatterMatrixData(int world_rank,
                                                                      const std::vector<int> &send_counts,
                                                                      const std::vector<int> &displacements) {
  local_flat_data_.resize(static_cast<size_t>(local_rows_) * cols_);
  MPI_Scatterv(world_rank == 0 ? input_matrix_.data() : nullptr, send_counts.data(), displacements.data(), MPI_INT,
               local_flat_data_.data(), static_cast<int>(local_flat_data_.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

std::vector<int> TsibarevaERibbonHorizontalMatrixMultVectorMPI::CalculateMultiplyLocalPart() {
  std::vector<int> local_result(local_rows_, 0);

  for (int local_row = 0; local_row < local_rows_; local_row++) {
    int sum = 0;
    for (int col = 0; col < cols_; col++) {
      int matrix_idx = (local_row * cols_) + col;
      sum += local_flat_data_[matrix_idx] * local_vector_[col];
    }
    local_result[local_row] = sum;
  }

  return local_result;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector
