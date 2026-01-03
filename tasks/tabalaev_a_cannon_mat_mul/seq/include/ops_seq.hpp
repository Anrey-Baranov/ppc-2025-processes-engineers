#pragma once

#include "tabalaev_a_cannon_mat_mul/common/include/common.hpp"
#include "task/include/task.hpp"

namespace tabalaev_a_cannon_mat_mul {

class TabalaevACannonMatMulSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TabalaevACannonMatMulSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tabalaev_a_cannon_mat_mul
