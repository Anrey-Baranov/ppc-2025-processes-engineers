#include "pankov_matrix_vector/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "pankov_matrix_vector/common/include/common.hpp"

namespace pankov_matrix_vector {

PankovMatrixVectorSEQ::PankovMatrixVectorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp(in);
  std::swap(GetInput(), temp);
  GetOutput() = std::vector<double>();
}

bool PankovMatrixVectorSEQ::ValidationImpl() {
  return GetOutput().empty();
}

bool PankovMatrixVectorSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  const std::size_t rows = input.matrix.size();
  GetOutput() = std::vector<double>(rows, 0.0);
  return true;
}

bool PankovMatrixVectorSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto &matrix = input.matrix;
  const auto &vector = input.vector;
  auto &result = GetOutput();

  const std::size_t rows = matrix.size();
  if (rows == 0) {
    return true;
  }

  const std::size_t cols = matrix[0].size();
  if (cols != vector.size()) {
    return false;
  }

  for (std::size_t i = 0; i < rows; ++i) {
    result[i] = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
      result[i] += matrix[i][j] * vector[j];
    }
  }

  return true;
}

bool PankovMatrixVectorSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace pankov_matrix_vector
