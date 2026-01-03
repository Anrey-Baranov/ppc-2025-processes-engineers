#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"

namespace vasiliev_m_bellman_ford_crs {

class VasilievMBellmanFordCrsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VasilievMBellmanFordCrsMPI(const InType &in);
  static void CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs);
  static int RelaxLocalEdges(int start, int end, const InType &in, const std::vector<int> &dist,
                             std::vector<int> &local_dist);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_bellman_ford_crs
