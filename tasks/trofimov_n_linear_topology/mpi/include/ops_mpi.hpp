#pragma once

#include "task/include/task.hpp"
#include "trofimov_n_linear_topology/common/include/common.hpp"

namespace trofimov_n_linear_topology {

class TrofimovNLinearTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TrofimovNLinearTopologyMPI(const InType &in);

 private:
  int rank_ = 0;
  int size_ = 0;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace trofimov_n_linear_topology
