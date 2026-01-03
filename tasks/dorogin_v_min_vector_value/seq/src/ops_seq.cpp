#include "dorogin_v_min_vector_value/seq/include/ops_seq.hpp"

#include <algorithm>

#include "dorogin_v_min_vector_value/common/include/common.hpp"

namespace dorogin_v_min_vector_value {

DoroginVMinVectorValueSEQ::DoroginVMinVectorValueSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DoroginVMinVectorValueSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool DoroginVMinVectorValueSEQ::PreProcessingImpl() {
  return true;
}

bool DoroginVMinVectorValueSEQ::RunImpl() {
  const auto &data = GetInput();

  const auto it_min = std::ranges::min_element(data);

  GetOutput() = (it_min != data.end()) ? *it_min : 0;

  return true;
}

bool DoroginVMinVectorValueSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dorogin_v_min_vector_value
