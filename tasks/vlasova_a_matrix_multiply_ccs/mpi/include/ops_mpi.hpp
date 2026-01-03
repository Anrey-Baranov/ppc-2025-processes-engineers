#pragma once

#include <utility>
#include <vector>

#include "task/include/task.hpp"
#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"

namespace vlasova_a_matrix_multiply_ccs {

class VlasovaAMatrixMultiplyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VlasovaAMatrixMultiplyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void TransposeMatrixMPI(const SparseMatrixCCS &a, SparseMatrixCCS &at);
  static std::pair<int, int> SplitColumns(int total_cols, int rank, int size);
  static void ExtractLocalColumns(const SparseMatrixCCS &b, int start_col, int end_col, std::vector<double> &loc_val,
                                  std::vector<int> &loc_row_ind, std::vector<int> &loc_col_ptr);
  static void MultiplyLocalMatrices(const SparseMatrixCCS &at, const std::vector<double> &loc_val,
                                    const std::vector<int> &loc_row_ind, const std::vector<int> &loc_col_ptr,
                                    int loc_cols, std::vector<double> &res_val, std::vector<int> &res_row_ind,
                                    std::vector<int> &res_col_ptr);
  bool ProcessRootRank(const SparseMatrixCCS &a, const SparseMatrixCCS &b, std::vector<double> &loc_res_val,
                       std::vector<int> &loc_res_row_ind, std::vector<int> &loc_res_col_ptr, int size);
  static bool ProcessWorkerRank(const std::vector<double> &loc_res_val, const std::vector<int> &loc_res_row_ind,
                                const std::vector<int> &loc_res_col_ptr, int loc_cols);
  static void ProcessLocalColumn(const SparseMatrixCCS &at, const std::vector<double> &loc_val,
                                 const std::vector<int> &loc_row_ind, const std::vector<int> &loc_col_ptr,
                                 int col_index, std::vector<double> &temp_row, std::vector<int> &row_marker,
                                 std::vector<double> &res_val, std::vector<int> &res_row_ind);
};

}  // namespace vlasova_a_matrix_multiply_ccs
