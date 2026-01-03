#include "likhanov_m_elem_vec_sum/seq/include/ops_seq.hpp"

#include <cstdint>

#include "likhanov_m_elem_vec_sum/common/include/common.hpp"

namespace likhanov_m_elem_vec_sum {

LikhanovMElemVecSumSEQ::LikhanovMElemVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LikhanovMElemVecSumSEQ::ValidationImpl() {
  return GetInput() >= 0;
}

bool LikhanovMElemVecSumSEQ::PreProcessingImpl() {
  return true;
}

bool LikhanovMElemVecSumSEQ::RunImpl() {
  const int64_t n = GetInput();

  int64_t sum = 0;
  for (int64_t i = 1; i <= n; ++i) {
    sum += i;
  }

  GetOutput() = sum;
  return true;
}

bool LikhanovMElemVecSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace likhanov_m_elem_vec_sum
