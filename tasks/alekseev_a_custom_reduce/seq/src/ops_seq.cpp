#include "alekseev_a_custom_reduce/seq/include/ops_seq.hpp"

#include "alekseev_a_custom_reduce/common/include/common.hpp"

namespace alekseev_a_custom_reduce {

AlekseevACustomReduceSEQ::AlekseevACustomReduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool AlekseevACustomReduceSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.root < 0) {
    return false;
  }
  if (input.data.empty()) {
    return false;
  }
  return true;
}

bool AlekseevACustomReduceSEQ::PreProcessingImpl() {
  return true;
}

bool AlekseevACustomReduceSEQ::RunImpl() {
  const auto &input = GetInput().data;
  double result = 0.0;
  int sz = static_cast<int>(input.size());
  for (auto i = 0; i < sz; i++) {
    result += input[i];
  }
  GetOutput() = result;
  return true;
}

bool AlekseevACustomReduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace alekseev_a_custom_reduce
