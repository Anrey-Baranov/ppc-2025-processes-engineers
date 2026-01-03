#pragma once

#include <vector>

#include "tabalaev_a_cannon_mat_mul/common/include/common.hpp"
#include "task/include/task.hpp"

namespace tabalaev_a_cannon_mat_mul {

class TabalaevACannonMatMulMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TabalaevACannonMatMulMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void LocalMatrixMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                                  int n);
};

}  // namespace tabalaev_a_cannon_mat_mul
