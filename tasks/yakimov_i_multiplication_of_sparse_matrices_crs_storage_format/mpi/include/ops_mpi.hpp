#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"
#include "yakimov_i_multiplication_of_sparse_matrices_crs_storage_format/common/include/common.hpp"

namespace yakimov_i_multiplication_of_sparse_matrices_crs_storage_format {

class YakimovIMultiplicationOfSparseMatricesMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit YakimovIMultiplicationOfSparseMatricesMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BroadcastMatrixInfo();
  void DistributeMatrixA();
  void DistributeMatrixB();
  void GatherResults();

  MatrixCRS matrix_A_;
  MatrixCRS matrix_B_;
  MatrixCRS result_matrix_;
  MatrixCRS local_a_rows_;
  MatrixCRS local_result_;

  std::string matrix_A_filename_;
  std::string matrix_B_filename_;
  std::vector<int> local_rows_;

  int rows_A_ = 0;
  int cols_A_ = 0;
  int rows_B_ = 0;
  int cols_B_ = 0;
};

}  // namespace yakimov_i_multiplication_of_sparse_matrices_crs_storage_format
