#pragma once

#include <vector>

#include "tabalaev_a_linear_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace tabalaev_a_linear_topology {

class TabalaevALinearTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TabalaevALinearTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] static int GetLeft(int rank);
  [[nodiscard]] static int GetRight(int rank, int size);
  [[nodiscard]] static int GetDirection(int sender, int receiver);
  [[nodiscard]] static bool IsOnPath(int rank, int sender, int receiver, int direction);
  void ProcessSender(int direction, int left, int right, std::vector<int> &local_buff);
  void static ProcessIntermediate(int direction, int left, int right, std::vector<int> &local_buff);
  void static ProcessReceiver(int direction, int left, int right, std::vector<int> &local_buff);
};

}  // namespace tabalaev_a_linear_topology
