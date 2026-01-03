#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/common/include/common.hpp"

namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector {

class TsibarevaERibbonHorizontalMatrixMultVectorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TsibarevaERibbonHorizontalMatrixMultVectorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BroadcastVector();
  void BroadcastMatrixDimensions();
  void PrepareScatterParameters(int world_rank, int world_size, std::vector<int> &send_counts,
                                std::vector<int> &displacements) const;
  void ScatterMatrixData(int world_rank, const std::vector<int> &send_counts, const std::vector<int> &displacements);
  std::vector<int> CalculateMultiplyLocalPart();

  std::vector<int> input_matrix_;

  std::vector<int> local_flat_data_;
  std::vector<int> local_vector_;

  int rows_ = 0;
  int cols_ = 0;
  int local_rows_ = 0;
};

}  // namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector
