#pragma once

#include "posternak_a_radix_merge_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace posternak_a_radix_merge_sort {

class PosternakARadixMergeSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PosternakARadixMergeSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace posternak_a_radix_merge_sort
