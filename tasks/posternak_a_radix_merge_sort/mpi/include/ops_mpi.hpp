#pragma once

#include <cstdint>
#include <vector>

#include "posternak_a_radix_merge_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace posternak_a_radix_merge_sort {

class PosternakARadixMergeSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PosternakARadixMergeSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<uint32_t> RadixSortLocal(std::vector<int> &data);
  static std::vector<int> ConvertToSigned(const std::vector<uint32_t> &unsigned_data);
  static void CalculateCountsAndOffsets(int input_len, int size, std::vector<int> &counts, std::vector<int> &offset);
  static void MergeSortedParts(std::vector<int> &result, const std::vector<std::vector<int>> &sorted_proc_parts);
};

}  // namespace posternak_a_radix_merge_sort
