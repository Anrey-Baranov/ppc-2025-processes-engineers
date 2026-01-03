#pragma once

#include <functional>
#include <vector>

#include "task/include/task.hpp"
#include "tochilin_e_integral_trapezium/common/include/common.hpp"

namespace tochilin_e_integral_trapezium {

class TochilinEIntegralTrapeziumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TochilinEIntegralTrapeziumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double ComputeIntegral();

  std::vector<double> lower_bounds_;
  std::vector<double> upper_bounds_;
  int num_steps_{};
  std::function<double(const std::vector<double> &)> func_;
  double result_{};
};

}  // namespace tochilin_e_integral_trapezium
