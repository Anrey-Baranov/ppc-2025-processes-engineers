#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "vasiliev_m_bubble_sort/common/include/common.hpp"

namespace vasiliev_m_bubble_sort {

class VasilievMBubbleSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VasilievMBubbleSortMPI(const InType &in);
  static void CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs);
  static void BubbleOESort(std::vector<int> &vec);
  static int FindPartner(int rank, int phase);
  static void ExchangeAndMerge(int rank, int partner, std::vector<int> &local_data);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_bubble_sort
