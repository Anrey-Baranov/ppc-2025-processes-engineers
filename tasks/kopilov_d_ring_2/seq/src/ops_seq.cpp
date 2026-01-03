#include "kopilov_d_ring_2/seq/include/ops_seq.hpp"

#include "kopilov_d_ring_2/common/include/common.hpp"

namespace kopilov_d_ring_2 {

KopilovDRingSEQ::KopilovDRingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KopilovDRingSEQ::ValidationImpl() {
  return true;
}

bool KopilovDRingSEQ::PreProcessingImpl() {
  GetOutput().data.clear();
  return true;
}

bool KopilovDRingSEQ::RunImpl() {
  GetOutput().data = GetInput().data;
  return true;
}

bool KopilovDRingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kopilov_d_ring_2
