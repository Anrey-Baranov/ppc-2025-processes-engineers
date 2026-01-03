#ifndef ZHURIN_I_RING_TOPOLOGY_MPI_INCLUDE_OPS_MPI_HPP_
#define ZHURIN_I_RING_TOPOLOGY_MPI_INCLUDE_OPS_MPI_HPP_

#include "task/include/task.hpp"
#include "zhurin_i_ring_topology/common/include/common.hpp"

namespace zhurin_i_ring_topology {

using BaseTask = ppc::task::Task<InType, OutType>;

class ZhurinIRingTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZhurinIRingTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zhurin_i_ring_topology

#endif  // ZHURIN_I_RING_TOPOLOGY_MPI_INCLUDE_OPS_MPI_HPP_
