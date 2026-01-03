#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "timofeev_n_radix_merge_sort/common/include/common.hpp"

namespace timofeev_n_radix_merge_sort {

class TimofeevNRadixMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TimofeevNRadixMergeMPI(const InType &in);

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
  static void SliyanieHelp(std::vector<std::vector<int>> &received, std::vector<int> &indexes, std::vector<int> &out,
                           int &i);
  static void Sliyanie(std::vector<std::vector<int>> &received, std::vector<int> &out);
  static void HandleTwoProcesses(std::vector<int> &in, std::vector<int> &out);
  static bool HandleZeroRank(std::vector<int> &in, std::vector<int> &out, int &size);
};

}  // namespace timofeev_n_radix_merge_sort
