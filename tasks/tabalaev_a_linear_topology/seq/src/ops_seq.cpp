#include "tabalaev_a_linear_topology/seq/include/ops_seq.hpp"

#include <vector>

#include "tabalaev_a_linear_topology/common/include/common.hpp"

namespace tabalaev_a_linear_topology {

TabalaevALinearTopologySEQ::TabalaevALinearTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TabalaevALinearTopologySEQ::ValidationImpl() {
  auto &sender = std::get<0>(GetInput());
  auto &receiver = std::get<1>(GetInput());
  auto &data = std::get<2>(GetInput());
  return sender >= 0 && receiver >= 0 && !data.empty();
}

bool TabalaevALinearTopologySEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool TabalaevALinearTopologySEQ::RunImpl() {
  auto &data = std::get<2>(GetInput());
  GetOutput() = data;
  return true;
}

bool TabalaevALinearTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_linear_topology
