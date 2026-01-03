#pragma once

#include <vector>

#include "bortsova_a_transmission_gather/common/include/common.hpp"
#include "task/include/task.hpp"

namespace bortsova_a_transmission_gather {

class BortsovaATransmissionGatherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BortsovaATransmissionGatherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void TreeGather(std::vector<double> &gather_buffer, std::vector<bool> &received, int local_count, int total_size);
  void ReceiveFromChild(std::vector<double> &gather_buffer, std::vector<bool> &received, int source, int local_count,
                        int total_size) const;
  void SendToParent(std::vector<double> &gather_buffer, std::vector<bool> &received, int step, int total_size) const;
  void TransferToRoot(std::vector<double> &gather_buffer, int root, int total_size) const;

  int world_rank_ = 0;
  int world_size_ = 1;
  int send_count_ = 0;
};

}  // namespace bortsova_a_transmission_gather
