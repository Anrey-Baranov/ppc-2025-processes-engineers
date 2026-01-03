#pragma once

#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

class BadanovASparseMatrixMultDoubleCcsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BadanovASparseMatrixMultDoubleCcsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct LocalData {
    SparseMatrix a_local;
    SparseMatrix b_local;
    int global_rows{};
    int global_inner_dim{};
    int global_cols{};
  };

  static LocalData DistributeDataHorizontal(int world_rank, int world_size, const SparseMatrix &a,
                                            const SparseMatrix &b);
  static SparseMatrix MultiplyLocal(const LocalData &local);
  static void GatherResults(int world_rank, int world_size, const SparseMatrix &local_c, SparseMatrix &global_c);
  static std::vector<double> SparseDotProduct(const SparseMatrix &a, const SparseMatrix &b, int col_b);
};

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
