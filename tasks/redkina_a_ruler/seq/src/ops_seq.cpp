#include "redkina_a_ruler/seq/include/ops_seq.hpp"

#include <vector>

#include "redkina_a_ruler/common/include/common.hpp"

namespace redkina_a_ruler {

RedkinaARulerSEQ::RedkinaARulerSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool RedkinaARulerSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.start >= 0 && input.end >= 0 && !input.data.empty();
}

bool RedkinaARulerSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool RedkinaARulerSEQ::RunImpl() {
  const auto &input = GetInput();
  GetOutput() = input.data;
  return true;
}

bool RedkinaARulerSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_ruler
