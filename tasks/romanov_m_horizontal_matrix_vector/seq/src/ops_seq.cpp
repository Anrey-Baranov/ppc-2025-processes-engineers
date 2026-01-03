#include "romanov_m_horizontal_matrix_vector/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

#include "romanov_m_horizontal_matrix_vector/common/include/common.hpp"

namespace romanov_m_horizontal_matrix_vector {

RomanovMHorizontalMatrixVectorSEQ::RomanovMHorizontalMatrixVectorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool RomanovMHorizontalMatrixVectorSEQ::ValidationImpl() {
  const auto &mat = std::get<0>(GetInput());
  const int r = std::get<1>(GetInput());
  const int c = std::get<2>(GetInput());
  const auto &v = std::get<3>(GetInput());

  if (r <= 0 || c <= 0) {
    return false;
  }

  const std::size_t expected_size = static_cast<std::size_t>(r) * static_cast<std::size_t>(c);

  if (mat.size() != expected_size) {
    return false;
  }
  if (v.size() != static_cast<std::size_t>(c)) {
    return false;
  }

  return true;
}

bool RomanovMHorizontalMatrixVectorSEQ::PreProcessingImpl() {
  GetOutput().resize(static_cast<std::size_t>(std::get<1>(GetInput())));
  return true;
}

bool RomanovMHorizontalMatrixVectorSEQ::RunImpl() {
  const auto &matrix = std::get<0>(GetInput());
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  const auto &vec = std::get<3>(GetInput());

  auto &res = GetOutput();

  for (int i = 0; i < rows; ++i) {
    double temp = 0.0;
    for (int j = 0; j < cols; ++j) {
      const std::size_t idx =
          (static_cast<std::size_t>(i) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(j);
      temp += matrix[idx] * vec[static_cast<std::size_t>(j)];
    }
    res[static_cast<std::size_t>(i)] = temp;
  }

  return true;
}

bool RomanovMHorizontalMatrixVectorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_m_horizontal_matrix_vector
