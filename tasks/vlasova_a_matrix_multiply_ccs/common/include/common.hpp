#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace vlasova_a_matrix_multiply_ccs {

struct SparseMatrixCCS {
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_ptrs;
  int rows = 0;
  int cols = 0;
  int nnz = 0;
};

using InType = std::pair<SparseMatrixCCS, SparseMatrixCCS>;
using OutType = SparseMatrixCCS;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

SparseMatrixCCS GenerateRandomSparseMatrix(int rows, int cols, double density);
bool CompareMatrices(const SparseMatrixCCS &a, const SparseMatrixCCS &b, double k_epsilon = 1e-6);

void TransposeMatrixCCS(const SparseMatrixCCS &a, SparseMatrixCCS &at);

}  // namespace vlasova_a_matrix_multiply_ccs
