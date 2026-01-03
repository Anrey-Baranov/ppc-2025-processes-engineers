#include "zhurin_i_ring_topology/seq/include/ops_seq.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "zhurin_i_ring_topology/common/include/common.hpp"

namespace zhurin_i_ring_topology {

ZhurinIRingTopologySEQ::ZhurinIRingTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ZhurinIRingTopologySEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.source >= 0 && input.dest >= 0;
}

bool ZhurinIRingTopologySEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ZhurinIRingTopologySEQ::RunImpl() {
  const auto &input = GetInput();
  GetOutput() = input.data;

  if (input.source != input.dest) {
    int distance = std::abs(input.dest - input.source);
    std::chrono::microseconds delay(static_cast<int64_t>(distance));
    std::this_thread::sleep_for(delay);
  }

  return true;
}

bool ZhurinIRingTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zhurin_i_ring_topology
