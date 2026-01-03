#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/common/include/common.hpp"

namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector {

class TsibarevaERibbonHorizontalMatrixMultVectorSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TsibarevaERibbonHorizontalMatrixMultVectorSEQ(const InType &in);

 private:
  std::vector<int> input_matrix_;  // входная матрица
  std::vector<int> input_vector_;  // входной вектор
  int rows_ = 0;
  int cols_ = 0;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector
