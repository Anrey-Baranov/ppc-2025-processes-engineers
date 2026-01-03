#pragma once

#include "balchunayte_z_shell_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace balchunayte_z_shell_batcher {

class BalchunayteZShellBatcherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit BalchunayteZShellBatcherSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType data_;
};

}  // namespace balchunayte_z_shell_batcher
