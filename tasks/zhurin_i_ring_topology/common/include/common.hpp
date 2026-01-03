#ifndef ZHURIN_I_RING_TOPOLOGY_COMMON_INCLUDE_COMMON_HPP_
#define ZHURIN_I_RING_TOPOLOGY_COMMON_INCLUDE_COMMON_HPP_

#include <tuple>
#include <vector>

namespace zhurin_i_ring_topology {

struct RingMessage {
  int source = 0;
  int dest = 0;
  std::vector<int> data;
  bool go_clockwise = true;
};

using InType = RingMessage;
using OutType = std::vector<int>;
using TestType = std::tuple<int, RingMessage>;

}  // namespace zhurin_i_ring_topology

#endif  // ZHURIN_I_RING_TOPOLOGY_COMMON_INCLUDE_COMMON_HPP_
