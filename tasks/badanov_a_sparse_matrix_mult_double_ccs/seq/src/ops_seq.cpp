#include "badanov_a_sparse_matrix_mult_double_ccs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

BadanovASparseMatrixMultDoubleCcsSEQ::BadanovASparseMatrixMultDoubleCcsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BadanovASparseMatrixMultDoubleCcsSEQ::ValidationImpl() {
  const auto &in = GetInput();

  const auto &values_a = std::get<0>(in);
  const auto &row_indices_a = std::get<1>(in);
  const auto &col_pointers_a = std::get<2>(in);
  const auto &value_b = std::get<3>(in);
  const auto &row_indices_b = std::get<4>(in);
  const auto &col_pointers_b = std::get<5>(in);
  int rows_a = std::get<6>(in);
  int cols_a = std::get<7>(in);
  int cols_b = std::get<8>(in);

  if (rows_a <= 0 || cols_a <= 0) {
    return false;
  }
  if (values_a.size() != row_indices_a.size()) {
    return false;
  }
  if (col_pointers_a.size() != static_cast<size_t>(cols_a) + 1) {
    return false;
  }

  if (cols_b <= 0) {
    return false;
  }
  if (value_b.size() != row_indices_b.size()) {
    return false;
  }
  if (col_pointers_b.size() != static_cast<size_t>(cols_b) + 1) {
    return false;
  }

  if (!std::ranges::all_of(row_indices_b, [cols_a](int row_idx) { return row_idx >= 0 && row_idx < cols_a; })) {
    return false;
  }

  if (!std::ranges::all_of(row_indices_a, [rows_a](int row_idx) { return row_idx >= 0 && row_idx < rows_a; })) {
    return false;
  }

  return true;
}

bool BadanovASparseMatrixMultDoubleCcsSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

double BadanovASparseMatrixMultDoubleCcsSEQ::DotProduct(const std::vector<double> &col_a,
                                                        const std::vector<double> &col_b) {
  double result = 0.0;
  for (size_t i = 0; i < col_a.size(); ++i) {
    result += col_a[i] * col_b[i];
  }
  return result;
}

SparseMatrix BadanovASparseMatrixMultDoubleCcsSEQ::MultiplyCCS(const SparseMatrix &a, const SparseMatrix &b) {
  std::vector<double> value_c;
  std::vector<int> row_indices_c;
  std::vector<int> col_pointers_c(b.cols + 1, 0);

  for (int j = 0; j < b.cols; ++j) {
    std::vector<double> temp_result(a.rows, 0.0);

    for (int idx_b = b.col_pointers[j]; idx_b < b.col_pointers[j + 1]; ++idx_b) {
      int row_b = b.row_indices[idx_b];
      double val_b = b.values[idx_b];

      for (int idx_a = a.col_pointers[row_b]; idx_a < a.col_pointers[row_b + 1]; ++idx_a) {
        int row_a = a.row_indices[idx_a];
        double val_a = a.values[idx_a];
        temp_result[row_a] += val_a * val_b;
      }
    }

    for (int i = 0; i < a.rows; ++i) {
      if (std::abs(temp_result[i]) > 1e-10) {
        value_c.push_back(temp_result[i]);
        row_indices_c.push_back(i);
        col_pointers_c[j + 1]++;
      }
    }
  }

  for (int j = 0; j < b.cols; ++j) {
    col_pointers_c[j + 1] += col_pointers_c[j];
  }

  SparseMatrix c;
  c.values = value_c;
  c.row_indices = row_indices_c;
  c.col_pointers = col_pointers_c;
  c.rows = a.rows;
  c.cols = b.cols;

  return c;
}

bool BadanovASparseMatrixMultDoubleCcsSEQ::RunImpl() {
  const auto &in = GetInput();

  const auto &values_a = std::get<0>(in);
  const auto &row_indices_a = std::get<1>(in);
  const auto &col_pointers_a = std::get<2>(in);
  const auto &value_b = std::get<3>(in);
  const auto &row_indices_b = std::get<4>(in);
  const auto &col_pointers_b = std::get<5>(in);
  int rows_a = std::get<6>(in);
  int cols_a = std::get<7>(in);
  int cols_b = std::get<8>(in);

  SparseMatrix a;
  a.values = values_a;
  a.row_indices = row_indices_a;
  a.col_pointers = col_pointers_a;
  a.rows = rows_a;
  a.cols = cols_a;

  SparseMatrix b;
  b.values = value_b;
  b.row_indices = row_indices_b;
  b.col_pointers = col_pointers_b;
  b.rows = cols_a;
  b.cols = cols_b;

  SparseMatrix c = MultiplyCCS(a, b);

  GetOutput() = std::make_tuple(c.values, c.row_indices, c.col_pointers);

  return true;
}

bool BadanovASparseMatrixMultDoubleCcsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
