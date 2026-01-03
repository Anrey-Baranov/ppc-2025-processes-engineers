#pragma once

#include "buzulukskiy_d_sort_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace buzulukskiy_d_sort_batcher {

class BuzulukskiyDSortBatcherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit BuzulukskiyDSortBatcherSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace buzulukskiy_d_sort_batcher
