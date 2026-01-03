#pragma once

#include "task/include/task.hpp"
#include "trofimov_n_linear_topology/common/include/common.hpp"

namespace trofimov_n_linear_topology {

class TrofimovNLinearTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TrofimovNLinearTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace trofimov_n_linear_topology
