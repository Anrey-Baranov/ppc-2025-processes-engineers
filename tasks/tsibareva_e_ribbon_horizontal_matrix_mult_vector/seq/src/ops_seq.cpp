#include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/common/include/common.hpp"

namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector {

TsibarevaERibbonHorizontalMatrixMultVectorSEQ::TsibarevaERibbonHorizontalMatrixMultVectorSEQ(const InType &in)
    : rows_(std::get<1>(in)), cols_(std::get<2>(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorSEQ::ValidationImpl() {
  return true;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorSEQ::PreProcessingImpl() {
  if (rows_ == 0 || cols_ == 0) {
    GetOutput() = std::vector<int>();
  } else {
    GetOutput() = std::vector<int>(static_cast<size_t>(rows_), 0);
  }
  return true;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorSEQ::RunImpl() {
  const auto &flat_matrix = std::get<0>(GetInput());
  const auto &flat_vector = std::get<3>(GetInput());

  input_matrix_ = std::vector<int>(flat_matrix);
  input_vector_ = std::vector<int>(flat_vector);

  auto &result_vector = GetOutput();

  for (int row = 0; row < rows_; ++row) {
    int sum = 0;
    for (int col = 0; col < cols_; ++col) {
      int matrix_idx = (row * cols_) + col;
      sum += input_matrix_[static_cast<size_t>(matrix_idx)] * input_vector_[static_cast<size_t>(col)];
    }
    result_vector[row] = sum;
  }

  return true;
}

bool TsibarevaERibbonHorizontalMatrixMultVectorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector
