#include "kondrashova_v_ring_topology/seq/include/ops_seq.hpp"

#include "kondrashova_v_ring_topology/common/include/common.hpp"

namespace kondrashova_v_ring_topology {

KondrashovaVRingTopologySEQ::KondrashovaVRingTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KondrashovaVRingTopologySEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.source >= 0 && input.recipient >= 0;
}

bool KondrashovaVRingTopologySEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool KondrashovaVRingTopologySEQ::RunImpl() {
  const auto &input = GetInput();
  result_ = input.data;
  return true;
}

bool KondrashovaVRingTopologySEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace kondrashova_v_ring_topology
