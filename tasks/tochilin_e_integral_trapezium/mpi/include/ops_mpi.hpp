#pragma once

#include <functional>
#include <vector>

#include "task/include/task.hpp"
#include "tochilin_e_integral_trapezium/common/include/common.hpp"

namespace tochilin_e_integral_trapezium {

class TochilinEIntegralTrapeziumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TochilinEIntegralTrapeziumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double ComputePartialIntegral(int start_idx, int end_idx);

  std::vector<double> lower_bounds_;
  std::vector<double> upper_bounds_;
  int num_steps_{};
  std::function<double(const std::vector<double> &)> func_;
  double result_{};
  int start_idx_{};
  int end_idx_{};
};

}  // namespace tochilin_e_integral_trapezium
