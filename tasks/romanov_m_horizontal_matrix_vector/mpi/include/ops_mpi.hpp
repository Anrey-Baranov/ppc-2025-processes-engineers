#pragma once

#include <vector>

#include "romanov_m_horizontal_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanov_m_horizontal_matrix_vector {

class RomanovMHorizontalMatrixVectorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RomanovMHorizontalMatrixVectorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CalculateDistribution(int rows, int proc_num, std::vector<int> &counts, std::vector<int> &displs);
};

}  // namespace romanov_m_horizontal_matrix_vector
