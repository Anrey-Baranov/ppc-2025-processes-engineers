#pragma once

#include "task/include/task.hpp"
#include "zaharov_g_seidel_int_met/common/include/common.hpp"

namespace zaharov_g_seidel_int_met {

class ZaharovGSeidelIntMetSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZaharovGSeidelIntMetSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  double epsilon_{0.0};
  int max_iterations_{0};
};

}  // namespace zaharov_g_seidel_int_met
