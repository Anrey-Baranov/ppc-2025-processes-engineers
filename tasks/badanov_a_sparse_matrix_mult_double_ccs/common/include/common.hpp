#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

using InType = std::tuple<std::vector<double>,  // values of matrix a
                          std::vector<int>,     // row indices of matrix a
                          std::vector<int>,     // column pointers of matrix a
                          std::vector<double>,  // values of matrix b
                          std::vector<int>,     // row indices of matrix b
                          std::vector<int>,     // column pointers of matrix b
                          int,                  // rows of a
                          int,                  // cols of a (rows of b)
                          int                   // cols of b
                          >;
using OutType = std::tuple<std::vector<double>,  // values of result matrix c
                           std::vector<int>,     // row indices of result matrix c
                           std::vector<int>      // column pointers of result matrix c
                           >;
using TestType = std::tuple<size_t, size_t, size_t>;  // rows, inner_dim, cols
using BaseTask = ppc::task::Task<InType, OutType>;

struct SparseMatrix {
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_pointers;
  int rows{};
  int cols{};
};

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
