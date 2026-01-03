#include "tsyplakov_k_from_all_to_one/seq/include/ops_seq.hpp"

namespace tsyplakov_k_from_all_to_one {

TsyplakovKFromAllToOneSEQ::TsyplakovKFromAllToOneSEQ(const InTypeSEQ &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
}

bool TsyplakovKFromAllToOneSEQ::ValidationImpl() {
  auto &[data, root] = this->GetInput();
  return !data.empty() && root >= 0;
}

bool TsyplakovKFromAllToOneSEQ::PreProcessingImpl() {
  gathered_.clear();
  return true;
}

bool TsyplakovKFromAllToOneSEQ::RunImpl() {
  auto &[data, root] = this->GetInput();

  if (root == 0) {
    this->GetOutput() = data;
  }

  return true;
}

bool TsyplakovKFromAllToOneSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_from_all_to_one
