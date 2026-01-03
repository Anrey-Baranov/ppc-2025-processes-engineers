#pragma once

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shemetov_d_increasing_contrast {

class IncreaseContrastTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit IncreaseContrastTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shemetov_d_increasing_contrast
