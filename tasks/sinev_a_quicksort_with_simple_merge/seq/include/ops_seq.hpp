#pragma once

#include <vector>

#include "sinev_a_quicksort_with_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sinev_a_quicksort_with_simple_merge {

class SinevAQuicksortWithSimpleMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SinevAQuicksortWithSimpleMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void QuickSortWithSimpleMerge(std::vector<int> &arr, int left, int right);
  static int Partition(std::vector<int> &arr, int left, int right);
  static void SimpleMerge(std::vector<int> &arr, int left, int mid, int right);
};

}  // namespace sinev_a_quicksort_with_simple_merge
