#include "pankov_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "pankov_matrix_vector/common/include/common.hpp"

namespace pankov_matrix_vector {

PankovMatrixVectorMPI::PankovMatrixVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp(in);
  std::swap(GetInput(), temp);
  GetOutput() = std::vector<double>();
}

bool PankovMatrixVectorMPI::ValidationImpl() {
  return GetOutput().empty();
}

bool PankovMatrixVectorMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  const std::size_t rows = input.matrix.size();
  GetOutput() = std::vector<double>(rows, 0.0);
  return true;
}

void PankovMatrixVectorMPI::DistributeDataFromRank0(const std::vector<std::vector<double>> &matrix,
                                                    std::vector<std::vector<double>> *local_matrix_band,
                                                    std::vector<double> *local_result, std::size_t u_rows,
                                                    std::size_t u_cols, std::size_t u_size, int size,
                                                    const std::vector<double> &local_vector) {
  std::size_t proc0_base_cols = u_cols / u_size;
  std::size_t proc0_rem_cols = u_cols % u_size;
  std::size_t proc0_start_col = 0;
  std::size_t proc0_end_col = proc0_base_cols + (std::cmp_less(0, static_cast<int>(proc0_rem_cols)) ? 1 : 0);
  proc0_end_col = std::min(proc0_end_col, u_cols);
  std::size_t proc0_local_cols = proc0_end_col - proc0_start_col;

  for (std::size_t i = 0; i < u_rows; ++i) {
    for (std::size_t j = 0; j < proc0_local_cols; ++j) {
      (*local_matrix_band)[i][j] = matrix[i][proc0_start_col + j];
    }
  }

  ComputePartialResults(*local_matrix_band, local_vector, local_result, u_rows, proc0_local_cols, proc0_start_col);

  std::vector<MPI_Request> send_requests;
  std::vector<std::vector<double>> send_buffers;
  send_requests.reserve(size - 1);
  send_buffers.reserve(size - 1);

  for (int proc = 1; proc < size; ++proc) {
    std::size_t proc_base_cols = u_cols / u_size;
    std::size_t proc_rem_cols = u_cols % u_size;
    std::size_t proc_start_col = (static_cast<std::size_t>(proc) * proc_base_cols) +
                                 static_cast<std::size_t>(std::min(proc, static_cast<int>(proc_rem_cols)));
    std::size_t proc_end_col =
        proc_start_col + proc_base_cols + (std::cmp_less(proc, static_cast<int>(proc_rem_cols)) ? 1 : 0);
    proc_end_col = std::min(proc_end_col, u_cols);
    std::size_t proc_local_cols = proc_end_col - proc_start_col;

    send_buffers.emplace_back(u_rows * proc_local_cols);
    std::vector<double> &send_buffer = send_buffers.back();

    for (std::size_t i = 0; i < u_rows; ++i) {
      for (std::size_t j = 0; j < proc_local_cols; ++j) {
        send_buffer[(i * proc_local_cols) + j] = matrix[i][proc_start_col + j];
      }
    }

    send_requests.emplace_back();
    MPI_Isend(send_buffer.data(), static_cast<int>(u_rows * proc_local_cols), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD,
              &send_requests.back());
  }

  if (!send_requests.empty()) {
    MPI_Waitall(static_cast<int>(send_requests.size()), send_requests.data(), MPI_STATUSES_IGNORE);
  }
}

void PankovMatrixVectorMPI::ReceiveDataOnRankNonZero(std::vector<std::vector<double>> *local_matrix_band,
                                                     std::size_t u_rows, std::size_t local_cols) {
  std::vector<double> recv_buffer(u_rows * local_cols);
  MPI_Request recv_request = MPI_REQUEST_NULL;
  MPI_Irecv(recv_buffer.data(), static_cast<int>(u_rows * local_cols), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &recv_request);

  MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

  for (std::size_t i = 0; i < u_rows; ++i) {
    for (std::size_t j = 0; j < local_cols; ++j) {
      (*local_matrix_band)[i][j] = recv_buffer[(i * local_cols) + j];
    }
  }
}

void PankovMatrixVectorMPI::ComputePartialResults(const std::vector<std::vector<double>> &local_matrix_band,
                                                  const std::vector<double> &local_vector,
                                                  std::vector<double> *local_result, std::size_t u_rows,
                                                  std::size_t local_cols, std::size_t start_col) {
  for (std::size_t i = 0; i < u_rows; ++i) {
    for (std::size_t j = 0; j < local_cols; ++j) {
      (*local_result)[i] += local_matrix_band[i][j] * local_vector[start_col + j];
    }
  }
}

bool PankovMatrixVectorMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const auto &matrix = input.matrix;
  const auto &vector = input.vector;

  int rows = 0;
  int cols = 0;
  int vector_size = 0;

  if (rank == 0) {
    rows = static_cast<int>(matrix.size());
    if (rows > 0) {
      cols = static_cast<int>(matrix[0].size());
    }
    vector_size = static_cast<int>(vector.size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&vector_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0 || cols == 0 || cols != vector_size) {
    GetOutput() = std::vector<double>(static_cast<std::size_t>(rows), 0.0);
    MPI_Barrier(MPI_COMM_WORLD);
    return cols == vector_size;
  }

  const auto u_cols = static_cast<std::size_t>(cols);
  const auto u_rows = static_cast<std::size_t>(rows);
  const auto u_size = static_cast<std::size_t>(size);

  std::size_t base_cols = u_cols / u_size;
  std::size_t rem_cols = u_cols % u_size;

  std::size_t start_col = (static_cast<std::size_t>(rank) * base_cols) +
                          static_cast<std::size_t>(std::min(rank, static_cast<int>(rem_cols)));
  std::size_t end_col = start_col + base_cols + (std::cmp_less(rank, static_cast<int>(rem_cols)) ? 1 : 0);
  end_col = std::min(end_col, u_cols);
  std::size_t local_cols = end_col - start_col;

  std::vector<double> local_vector(static_cast<std::size_t>(vector_size));
  if (rank == 0) {
    local_vector = vector;
  }
  MPI_Bcast(local_vector.data(), vector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<std::vector<double>> local_matrix_band(u_rows);
  for (std::size_t i = 0; i < u_rows; ++i) {
    local_matrix_band[i].resize(local_cols);
  }

  std::vector<double> local_result(u_rows, 0.0);

  if (rank == 0) {
    DistributeDataFromRank0(matrix, &local_matrix_band, &local_result, u_rows, u_cols, u_size, size, local_vector);
  } else {
    ReceiveDataOnRankNonZero(&local_matrix_band, u_rows, local_cols);
    ComputePartialResults(local_matrix_band, local_vector, &local_result, u_rows, local_cols, start_col);
  }

  std::vector<double> global_result(u_rows, 0.0);
  MPI_Reduce(local_result.data(), global_result.data(), rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(global_result.data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_result;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool PankovMatrixVectorMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace pankov_matrix_vector
