#pragma once

#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

class BadanovASparseMatrixMultDoubleCcsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BadanovASparseMatrixMultDoubleCcsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static SparseMatrix MultiplyCCS(const SparseMatrix &a, const SparseMatrix &b);
  static double DotProduct(const std::vector<double> &col_a, const std::vector<double> &col_b);
};

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
