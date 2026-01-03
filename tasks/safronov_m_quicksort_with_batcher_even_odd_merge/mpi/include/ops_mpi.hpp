#pragma once

#include <utility>
#include <vector>

#include "safronov_m_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace safronov_m_quicksort_with_batcher_even_odd_merge {

class SafronovMQuicksortWithBatcherEvenOddMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SafronovMQuicksortWithBatcherEvenOddMergeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static std::vector<int> CalculatingInterval(int size_prcs, int rank, int size_arr);
  void SendingVector(int rank);
  static void MergeAndSplit(std::vector<int> &own_data, std::vector<int> &neighbor_data, bool flag);
  static void DataExchange(std::vector<int> &own_data, int size_arr, int rank, int size, int neighbor);
  static void BatcherOddEvenPhases(std::vector<int> &own_data, std::vector<int> &interval, int size_arr, int rank,
                                   int size);
  static void EvenPhase(std::vector<int> &own_data, std::vector<int> &interval, int size_arr, int rank, int size);
  static void OddPhase(std::vector<int> &own_data, std::vector<int> &interval, int size_arr, int rank, int size);
  static int LengthsLocalArrays(int size_arr, int rank, int size);
  static void QuickSort(std::vector<int> &array);
  static std::pair<int, int> SplitRange(std::vector<int> &array, int left, int right);
  void SendingResult(int rank);
};

}  // namespace safronov_m_quicksort_with_batcher_even_odd_merge
