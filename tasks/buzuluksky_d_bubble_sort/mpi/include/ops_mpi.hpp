#pragma once

#include "buzuluksky_d_bubble_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace buzuluksky_d_bubble_sort {

class BuzulukskyDBubbleSortMPI final : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit BuzulukskyDBubbleSortMPI(const InType &input);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace buzuluksky_d_bubble_sort
