#pragma once

#include "buzulukskiy_d_sort_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace buzulukskiy_d_sort_batcher {

class BuzulukskiyDSortBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit BuzulukskiyDSortBatcherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace buzulukskiy_d_sort_batcher
