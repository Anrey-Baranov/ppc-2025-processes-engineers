#include "bortsova_a_transmission_gather/seq/include/ops_seq.hpp"

#include <vector>

#include "bortsova_a_transmission_gather/common/include/common.hpp"

namespace bortsova_a_transmission_gather {

BortsovaATransmissionGatherSEQ::BortsovaATransmissionGatherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool BortsovaATransmissionGatherSEQ::ValidationImpl() {
  int root = GetInput().root;
  return root == 0;
}

bool BortsovaATransmissionGatherSEQ::PreProcessingImpl() {
  return true;
}

bool BortsovaATransmissionGatherSEQ::RunImpl() {
  const std::vector<double> &send_data = GetInput().send_data;
  GetOutput().recv_data = send_data;
  return true;
}

bool BortsovaATransmissionGatherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace bortsova_a_transmission_gather
