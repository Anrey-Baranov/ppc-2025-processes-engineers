#pragma once

#include <cstddef>
#include <vector>

#include "pankov_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pankov_matrix_vector {

class PankovMatrixVectorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PankovMatrixVectorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void DistributeDataFromRank0(const std::vector<std::vector<double>> &matrix,
                                      std::vector<std::vector<double>> *local_matrix_band,
                                      std::vector<double> *local_result, std::size_t u_rows, std::size_t u_cols,
                                      std::size_t u_size, int size, const std::vector<double> &local_vector);
  static void ReceiveDataOnRankNonZero(std::vector<std::vector<double>> *local_matrix_band, std::size_t u_rows,
                                       std::size_t local_cols);
  static void ComputePartialResults(const std::vector<std::vector<double>> &local_matrix_band,
                                    const std::vector<double> &local_vector, std::vector<double> *local_result,
                                    std::size_t u_rows, std::size_t local_cols, std::size_t start_col);
};

}  // namespace pankov_matrix_vector
