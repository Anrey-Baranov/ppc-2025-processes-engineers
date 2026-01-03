#pragma once

#include <vector>

#include "afanasyev_a_it_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace afanasyev_a_it_seidel_method {

class AfanasyevAItSeidelMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AfanasyevAItSeidelMethodMPI(const InType &in);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  double epsilon_{0.0};
  int max_iterations_{0};
};

}  // namespace afanasyev_a_it_seidel_method
