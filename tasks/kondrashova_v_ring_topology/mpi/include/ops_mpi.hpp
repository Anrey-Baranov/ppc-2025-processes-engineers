#pragma once

#include <vector>

#include "kondrashova_v_ring_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondrashova_v_ring_topology {

class KondrashovaVRingTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KondrashovaVRingTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void SendData(int rank, int sender, int receiver, int step, int data_size, const std::vector<int> &data,
                       const std::vector<int> &buffer);
  void ReceiveData(int rank, int sender, int receiver, int recipient, std::vector<int> &buffer);
};

}  // namespace kondrashova_v_ring_topology
