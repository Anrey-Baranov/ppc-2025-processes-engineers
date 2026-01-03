#pragma once

#include <vector>

#include "karpich_i_seidol_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace karpich_i_seidol_method {

class KarpichISeidolMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KarpichISeidolMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static bool IterContinue(std::vector<double> &iter_eps, double correct_eps);
};
}  // namespace karpich_i_seidol_method
