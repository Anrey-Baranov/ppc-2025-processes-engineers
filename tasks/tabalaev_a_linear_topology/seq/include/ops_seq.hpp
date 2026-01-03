#pragma once

#include "tabalaev_a_linear_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace tabalaev_a_linear_topology {

class TabalaevALinearTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TabalaevALinearTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tabalaev_a_linear_topology
