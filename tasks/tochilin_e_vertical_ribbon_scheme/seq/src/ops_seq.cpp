#include "tochilin_e_vertical_ribbon_scheme/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "tochilin_e_vertical_ribbon_scheme/common/include/common.hpp"

namespace tochilin_e_vertical_ribbon_scheme {

TochilinEVerticalRibbonSchemeSEQ::TochilinEVerticalRibbonSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TochilinEVerticalRibbonSchemeSEQ::ValidationImpl() {
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
  return true;
}

bool TochilinEVerticalRibbonSchemeSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  rows_ = input.rows;
  cols_ = input.cols;
  matrix_ = input.matrix;
  vector_ = input.vector;
  result_.assign(rows_, 0.0);
  return true;
}

bool TochilinEVerticalRibbonSchemeSEQ::RunImpl() {
  for (int j = 0; j < cols_; j++) {
    for (int i = 0; i < rows_; i++) {
      result_[i] += matrix_[static_cast<std::size_t>(j * rows_) + i] * vector_[j];
    }
  }
  return true;
}

bool TochilinEVerticalRibbonSchemeSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace tochilin_e_vertical_ribbon_scheme
