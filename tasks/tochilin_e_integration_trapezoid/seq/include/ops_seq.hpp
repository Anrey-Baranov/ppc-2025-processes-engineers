#pragma once

#include "task/include/task.hpp"
#include "tochilin_e_integration_trapezoid/common/include/common.hpp"

namespace tochilin_e_integration_trapezoid {

class TochilinEIntegrationTrapezoidSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TochilinEIntegrationTrapezoidSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double lower_bound_{0.0};
  double upper_bound_{0.0};
  int num_intervals_{0};
  FunctionType function_;
  double result_{0.0};
};

}  // namespace tochilin_e_integration_trapezoid
