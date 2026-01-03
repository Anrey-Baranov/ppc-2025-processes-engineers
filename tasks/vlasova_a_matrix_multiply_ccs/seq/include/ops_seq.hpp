#pragma once

#include "task/include/task.hpp"
#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"

namespace vlasova_a_matrix_multiply_ccs {
class VlasovaAMatrixMultiplySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VlasovaAMatrixMultiplySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void TransposeMatrix(const SparseMatrixCCS &a, SparseMatrixCCS &at);
  static void MultiplyMatrices(const SparseMatrixCCS &a, const SparseMatrixCCS &b, SparseMatrixCCS &c);
};

}  // namespace vlasova_a_matrix_multiply_ccs
