#pragma once

#include "rastvorov_k_radix_sort_double_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rastvorov_k_radix_sort_double_merge {

class RastvorovKRadixSortDoubleMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit RastvorovKRadixSortDoubleMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType data_;
};

}  // namespace rastvorov_k_radix_sort_double_merge
