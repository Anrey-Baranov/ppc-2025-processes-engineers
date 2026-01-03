#pragma once

#include <vector>

#include "sinev_a_quicksort_with_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sinev_a_quicksort_with_simple_merge {

class SinevAQuicksortWithSimpleMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SinevAQuicksortWithSimpleMergeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int Partition(std::vector<int> &arr, int left, int right);
  static void SimpleMerge(std::vector<int> &arr, int left, int mid, int right);
  static void QuickSortWithSimpleMerge(std::vector<int> &arr, int left, int right);

  void PerformMultiWayMerge(const std::vector<int> &all_sizes);
  void BroadcastFinalResult();
  void ParallelQuickSort();
  std::vector<int> DistributeData(int world_size, int world_rank);

  struct DistributionInfo {
    std::vector<int> send_counts;
    std::vector<int> displacements;
    int local_size = 0;
  };

  static DistributionInfo PrepareDistributionInfo(int total_size, int world_size, int world_rank);
};

}  // namespace sinev_a_quicksort_with_simple_merge
