#include "sinev_a_allreduce/seq/include/ops_seq.hpp"

#include <exception>

#include "sinev_a_allreduce/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace sinev_a_allreduce {

SinevAAllreduceSEQ::SinevAAllreduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

bool SinevAAllreduceSEQ::ValidationImpl() {
  try {
    return true;
  } catch (...) {
    return false;
  }
}

bool SinevAAllreduceSEQ::PreProcessingImpl() {
  return true;
}

bool SinevAAllreduceSEQ::RunImpl() {
  try {
    GetOutput() = GetInput();
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool SinevAAllreduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_allreduce
