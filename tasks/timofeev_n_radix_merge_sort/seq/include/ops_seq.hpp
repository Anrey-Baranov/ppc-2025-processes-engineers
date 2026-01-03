#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "timofeev_n_radix_merge_sort/common/include/common.hpp"

namespace timofeev_n_radix_merge_sort {

class TimofeevNRadixMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TimofeevNRadixMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RadixMergeSort(std::vector<int> &part);
  static int GetDigit(int num, int digit);
  static int GetMaxDigits(const std::vector<int> &arr);
  static void SplitPosNeg(const std::vector<int> &arr, std::vector<int> &negative, std::vector<int> &positive);
  static void RadixMergeBucketHelpingFunction(std::vector<int> &part, int digit);
};

}  // namespace timofeev_n_radix_merge_sort
