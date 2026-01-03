#pragma once

#include "task/include/task.hpp"
#include "zaharov_g_seidel_int_met/common/include/common.hpp"

namespace zaharov_g_seidel_int_met {

class ZaharovGSeidelIntMetMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZaharovGSeidelIntMetMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  double CalculateMaxDiff(const std::vector<double> &a, const std::vector<double> &b, int start_idx, int end_idx);

  std::vector<std::vector<double>> local_A_;
  std::vector<double> local_b_;
  std::vector<double> x_;
  double epsilon_{0.0};
  int max_iterations_{0};
  int system_size_{0};
  int rows_per_process_{0};
  int remainder_{0};
  int start_row_{0};
  int end_row_{0};
  int local_rows_{0};
  int rank_{0};
  int size_{1};
};

}  // namespace zaharov_g_seidel_int_met
