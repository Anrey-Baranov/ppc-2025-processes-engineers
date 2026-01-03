#include "vlasova_a_matrix_multiply_ccs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"

namespace vlasova_a_matrix_multiply_ccs {

namespace {
constexpr double kEpsilon = 1e-10;
}  // namespace

VlasovaAMatrixMultiplyMPI::VlasovaAMatrixMultiplyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = SparseMatrixCCS{};
}

bool VlasovaAMatrixMultiplyMPI::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return (a.rows > 0 && a.cols > 0 && b.rows > 0 && b.cols > 0 && a.cols == b.rows);
}

bool VlasovaAMatrixMultiplyMPI::PreProcessingImpl() {
  return true;
}

void VlasovaAMatrixMultiplyMPI::TransposeMatrixMPI(const SparseMatrixCCS &a, SparseMatrixCCS &at) {
  at.rows = a.cols;
  at.cols = a.rows;
  at.nnz = a.nnz;

  if (a.nnz == 0) {
    at.values.clear();
    at.row_indices.clear();
    at.col_ptrs.assign(at.cols + 1, 0);
    return;
  }

  std::vector<int> row_count(at.cols, 0);
  for (int i = 0; i < a.nnz; i++) {
    row_count[a.row_indices[i]]++;
  }

  at.col_ptrs.resize(at.cols + 1);
  at.col_ptrs[0] = 0;
  for (int i = 0; i < at.cols; i++) {
    at.col_ptrs[i + 1] = at.col_ptrs[i] + row_count[i];
  }

  at.values.resize(a.nnz);
  at.row_indices.resize(a.nnz);

  std::vector<int> current_pos(at.cols, 0);
  for (int col = 0; col < a.cols; col++) {
    for (int i = a.col_ptrs[col]; i < a.col_ptrs[col + 1]; i++) {
      int row = a.row_indices[i];
      double val = a.values[i];

      int pos = at.col_ptrs[row] + current_pos[row];
      at.values[pos] = val;
      at.row_indices[pos] = col;
      current_pos[row]++;
    }
  }
}

std::pair<int, int> VlasovaAMatrixMultiplyMPI::SplitColumns(int total_cols, int rank, int size) {
  int base_cols = total_cols / size;
  int remainder = total_cols % size;

  int start_col = (rank * base_cols) + std::min(rank, remainder);
  int end_col = start_col + base_cols + (rank < remainder ? 1 : 0);

  return {start_col, end_col};
}

void VlasovaAMatrixMultiplyMPI::ProcessLocalColumn(const SparseMatrixCCS &at, const std::vector<double> &loc_val,
                                                   const std::vector<int> &loc_row_ind,
                                                   const std::vector<int> &loc_col_ptr, int col_index,
                                                   std::vector<double> &temp_row, std::vector<int> &row_marker,
                                                   std::vector<double> &res_val, std::vector<int> &res_row_ind) {
  int col_start = loc_col_ptr[col_index];
  int col_end = loc_col_ptr[col_index + 1];

  for (int k = col_start; k < col_end; k++) {
    int row_b = loc_row_ind[k];
    double val_b = loc_val[k];

    for (int idx = at.col_ptrs[row_b]; idx < at.col_ptrs[row_b + 1]; idx++) {
      int row_a = at.row_indices[idx];
      double val_a = at.values[idx];

      if (row_marker[row_a] != col_index) {
        row_marker[row_a] = col_index;
        temp_row[row_a] = val_a * val_b;
      } else {
        temp_row[row_a] += val_a * val_b;
      }
    }
  }

  for (int i = 0; i < at.cols; i++) {
    if (row_marker[i] == col_index && std::abs(temp_row[i]) > kEpsilon) {
      res_val.push_back(temp_row[i]);
      res_row_ind.push_back(i);
    }
  }
}

void VlasovaAMatrixMultiplyMPI::ExtractLocalColumns(const SparseMatrixCCS &b, int start_col, int end_col,
                                                    std::vector<double> &loc_val, std::vector<int> &loc_row_ind,
                                                    std::vector<int> &loc_col_ptr) {
  loc_val.clear();
  loc_row_ind.clear();
  loc_col_ptr.clear();

  loc_col_ptr.push_back(0);

  for (int col = start_col; col < end_col; col++) {
    int start_index = b.col_ptrs[col];
    int end_index = b.col_ptrs[col + 1];

    for (int i = start_index; i < end_index; i++) {
      loc_val.push_back(b.values[i]);
      loc_row_ind.push_back(b.row_indices[i]);
    }

    loc_col_ptr.push_back(static_cast<int>(loc_val.size()));
  }
}

void VlasovaAMatrixMultiplyMPI::MultiplyLocalMatrices(const SparseMatrixCCS &at, const std::vector<double> &loc_val,
                                                      const std::vector<int> &loc_row_ind,
                                                      const std::vector<int> &loc_col_ptr, int loc_cols,
                                                      std::vector<double> &res_val, std::vector<int> &res_row_ind,
                                                      std::vector<int> &res_col_ptr) {
  res_val.clear();
  res_row_ind.clear();
  res_col_ptr.clear();
  res_col_ptr.push_back(0);

  std::vector<double> temp_row(at.cols, 0.0);
  std::vector<int> row_marker(at.cols, -1);

  for (int j = 0; j < loc_cols; j++) {
    ProcessLocalColumn(at, loc_val, loc_row_ind, loc_col_ptr, j, temp_row, row_marker, res_val, res_row_ind);
    res_col_ptr.push_back(static_cast<int>(res_val.size()));
  }
}

bool VlasovaAMatrixMultiplyMPI::ProcessRootRank(const SparseMatrixCCS &a, const SparseMatrixCCS &b,
                                                std::vector<double> &loc_res_val, std::vector<int> &loc_res_row_ind,
                                                std::vector<int> &loc_res_col_ptr, int size) {
  SparseMatrixCCS c;
  c.rows = a.rows;
  c.cols = b.cols;

  std::vector<std::vector<double>> all_values(size);
  std::vector<std::vector<int>> all_row_indices(size);
  std::vector<std::vector<int>> all_col_ptrs(size);

  all_values[0] = std::move(loc_res_val);
  all_row_indices[0] = std::move(loc_res_row_ind);
  all_col_ptrs[0] = std::move(loc_res_col_ptr);

  for (int src = 1; src < size; src++) {
    int src_nnz = 0;
    int src_cols = 0;
    MPI_Recv(&src_nnz, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&src_cols, 1, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<double> src_vals(src_nnz);
    std::vector<int> src_rows(src_nnz);
    std::vector<int> src_ptrs(src_cols + 1);

    MPI_Recv(src_vals.data(), src_nnz, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(src_rows.data(), src_nnz, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(src_ptrs.data(), src_cols + 1, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    all_values[src] = std::move(src_vals);
    all_row_indices[src] = std::move(src_rows);
    all_col_ptrs[src] = std::move(src_ptrs);
  }

  c.col_ptrs.push_back(0);

  std::vector<int> value_offsets(size, 0);
  std::vector<int> col_offsets(size, 0);

  for (int i = 0; i < size; i++) {
    if (i > 0) {
      value_offsets[i] = value_offsets[i - 1] + static_cast<int>(all_values[i - 1].size());
      col_offsets[i] = col_offsets[i - 1] + static_cast<int>(all_col_ptrs[i - 1].size() - 1);
    }
  }

  for (int i = 0; i < size; i++) {
    c.values.insert(c.values.end(), all_values[i].begin(), all_values[i].end());
    c.row_indices.insert(c.row_indices.end(), all_row_indices[i].begin(), all_row_indices[i].end());

    for (size_t j = 1; j < all_col_ptrs[i].size(); j++) {
      c.col_ptrs.push_back(all_col_ptrs[i][j] + value_offsets[i]);
    }
  }

  c.nnz = static_cast<int>(c.values.size());
  GetOutput() = c;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool VlasovaAMatrixMultiplyMPI::ProcessWorkerRank(const std::vector<double> &loc_res_val,
                                                  const std::vector<int> &loc_res_row_ind,
                                                  const std::vector<int> &loc_res_col_ptr, int loc_cols) {
  int local_nnz = static_cast<int>(loc_res_val.size());
  int local_cols = loc_cols;

  MPI_Send(&local_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  MPI_Send(&local_cols, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
  MPI_Send(loc_res_val.data(), local_nnz, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  MPI_Send(loc_res_row_ind.data(), local_nnz, MPI_INT, 0, 3, MPI_COMM_WORLD);
  MPI_Send(loc_res_col_ptr.data(), loc_cols + 1, MPI_INT, 0, 4, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool VlasovaAMatrixMultiplyMPI::RunImpl() {
  const auto &[a, b] = GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  SparseMatrixCCS at;
  if (rank == 0) {
    TransposeMatrixMPI(a, at);
  } else {
    at.rows = a.cols;
    at.cols = a.rows;
  }

  MPI_Bcast(&at.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&at.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    at.nnz = static_cast<int>(at.values.size());
  }
  MPI_Bcast(&at.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    at.values.resize(at.nnz);
    at.row_indices.resize(at.nnz);
    at.col_ptrs.resize(at.cols + 1);
  }

  MPI_Bcast(at.values.data(), at.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(at.row_indices.data(), at.nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(at.col_ptrs.data(), at.cols + 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto [start_col, end_col] = SplitColumns(b.cols, rank, size);
  int loc_cols = end_col - start_col;

  std::vector<double> loc_b_val;
  std::vector<int> loc_b_row_ind;
  std::vector<int> loc_b_col_ptr;

  ExtractLocalColumns(b, start_col, end_col, loc_b_val, loc_b_row_ind, loc_b_col_ptr);

  std::vector<double> loc_res_val;
  std::vector<int> loc_res_row_ind;
  std::vector<int> loc_res_col_ptr;

  MultiplyLocalMatrices(at, loc_b_val, loc_b_row_ind, loc_b_col_ptr, loc_cols, loc_res_val, loc_res_row_ind,
                        loc_res_col_ptr);

  if (rank == 0) {
    return ProcessRootRank(a, b, loc_res_val, loc_res_row_ind, loc_res_col_ptr, size);
  }

  return ProcessWorkerRank(loc_res_val, loc_res_row_ind, loc_res_col_ptr, loc_cols);
}

bool VlasovaAMatrixMultiplyMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &c = GetOutput();

  if (rank == 0) {
    return c.rows > 0 && c.cols > 0 && c.col_ptrs.size() == static_cast<size_t>(c.cols) + 1;
  }

  return c.rows == 0 && c.cols == 0;
}

}  // namespace vlasova_a_matrix_multiply_ccs
