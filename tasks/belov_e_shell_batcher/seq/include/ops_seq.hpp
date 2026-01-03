#pragma once

#include "belov_e_shell_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace belov_e_shell_batcher {
class BelovEShellBatcherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BelovEShellBatcherSEQ(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace belov_e_shell_batcher
