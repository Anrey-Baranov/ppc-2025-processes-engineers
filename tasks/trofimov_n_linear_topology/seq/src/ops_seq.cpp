#include "trofimov_n_linear_topology/seq/include/ops_seq.hpp"

#include "trofimov_n_linear_topology/common/include/common.hpp"

namespace trofimov_n_linear_topology {

TrofimovNLinearTopologySEQ::TrofimovNLinearTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TrofimovNLinearTopologySEQ::ValidationImpl() {
  const auto &in = GetInput();
  return in.source >= 0 && in.target >= 0;
}

bool TrofimovNLinearTopologySEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TrofimovNLinearTopologySEQ::RunImpl() {
  const auto &in = GetInput();

  Work(in.value);

  GetOutput() = in.value;
  return true;
}

bool TrofimovNLinearTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_linear_topology
