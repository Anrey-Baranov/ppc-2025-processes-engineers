#pragma once

#include <string>

#include "task/include/task.hpp"
#include "yakimov_i_multiplication_of_sparse_matrices_crs_storage_format/common/include/common.hpp"

namespace yakimov_i_multiplication_of_sparse_matrices_crs_storage_format {

class YakimovIMultiplicationOfSparseMatricesSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit YakimovIMultiplicationOfSparseMatricesSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  MatrixCRS matrix_A_;
  MatrixCRS matrix_B_;
  MatrixCRS result_matrix_;
  std::string matrix_A_filename_;
  std::string matrix_B_filename_;
};

}  // namespace yakimov_i_multiplication_of_sparse_matrices_crs_storage_format
