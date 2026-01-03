#pragma once

#include <vector>

#include "shkryleva_s_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkryleva_s_seidel_method {

class ShkrylevaSSeidelMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShkrylevaSSeidelMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ComputeRowDistribution(int n, int size, std::vector<int> &row_counts, std::vector<int> &row_displs,
                                     std::vector<int> &matrix_counts, std::vector<int> &matrix_displs);
  static void InitializeMatrixAndVector(std::vector<double> &flat_matrix, std::vector<double> &b, int n);
  static bool SolveIteratively(int local_rows, int start_row, int n, const std::vector<double> &local_matrix,
                               const std::vector<double> &local_b, std::vector<double> &x,
                               const std::vector<int> &row_counts, const std::vector<int> &row_displs, double epsilon,
                               int max_iterations);

  static double PerformLocalIteration(int local_rows, int start_row, int n, const std::vector<double> &local_matrix,
                                      const std::vector<double> &local_b, std::vector<double> &x);
  static void GatherX(int local_rows, int start_row, std::vector<double> &x, const std::vector<int> &row_counts,
                      const std::vector<int> &row_displs);
};

}  // namespace shkryleva_s_seidel_method
