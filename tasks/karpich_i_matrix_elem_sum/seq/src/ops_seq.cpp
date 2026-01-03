#include "karpich_i_matrix_elem_sum/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "karpich_i_matrix_elem_sum/common/include/common.hpp"

namespace karpich_i_matrix_elem_sum {
KarpichIMatrixElemSumSEQ::KarpichIMatrixElemSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KarpichIMatrixElemSumSEQ::ValidationImpl() {
  std::size_t n = std::get<0>(GetInput());
  std::size_t m = std::get<1>(GetInput());
  std::vector<int> &val = std::get<2>(GetInput());

  return (n > 0) && (m > 0) && (val.size() == (n * m));
}

bool KarpichIMatrixElemSumSEQ::PreProcessingImpl() {
  return true;
}

bool KarpichIMatrixElemSumSEQ::RunImpl() {
  std::size_t n = std::get<0>(GetInput());
  std::size_t m = std::get<1>(GetInput());
  std::vector<int> &val = std::get<2>(GetInput());
  if (n == 0 || m == 0 || val.size() != (n * m)) {
    return false;
  }

  std::int64_t sum = 0;
  for (int v : val) {
    sum += v;
  }
  GetOutput() = sum;
  return true;
}

bool KarpichIMatrixElemSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace karpich_i_matrix_elem_sum
