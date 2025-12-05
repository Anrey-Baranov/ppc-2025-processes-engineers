#include "baranov_a_custom_allreduce/seq/include/ops_seq.hpp"

#include <iostream>
#include <stdexcept>
#include <variant>
#include <vector>

#include "baranov_a_custom_allreduce/common/include/common.hpp"

namespace baranov_a_custom_allreduce {

BaranovACustomAllreduceSEQ::BaranovACustomAllreduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

bool BaranovACustomAllreduceSEQ::ValidationImpl() {
  try {
    auto output = GetOutput();
    std::visit([](const auto &vec) {
      for (const auto &val : vec) {
        if (std::isnan(val) || std::isinf(val)) {
          throw std::runtime_error("Invalid value in output");
        }
      }
    }, output);
    return true;
  } catch (...) {
    return false;
  }
}

bool BaranovACustomAllreduceSEQ::PreProcessingImpl() {
  return true;
}

bool BaranovACustomAllreduceSEQ::RunImpl() {
  try {
    GetOutput() = GetInput();
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool BaranovACustomAllreduceSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace baranov_a_custom_allreduce
