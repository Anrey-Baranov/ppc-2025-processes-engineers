#include "golovanov_d_bcast/seq/include/ops_seq.hpp"

#include "golovanov_d_bcast/common/include/common.hpp"
// #include "util/include/util.hpp" clang-tidy попросил отключить

namespace golovanov_d_bcast {

GolovanovDBcastSEQ::GolovanovDBcastSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = false;
}

bool GolovanovDBcastSEQ::ValidationImpl() {
  return true;
}

bool GolovanovDBcastSEQ::PreProcessingImpl() {
  return true;
}
// заглушка
bool GolovanovDBcastSEQ::RunImpl() {
  GetOutput() = true;
  return true;
}

bool GolovanovDBcastSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace golovanov_d_bcast
