#pragma once

#include <cstddef>
#include <vector>

#include "shilin_n_gauss_band_horizontal_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shilin_n_gauss_band_horizontal_scheme {

class ShilinNGaussBandHorizontalSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShilinNGaussBandHorizontalSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int ValidateInput(const InType &input);
  static void DistributeRows(const InType &augmented_matrix, size_t n, size_t cols, int rank, int size,
                             InType &local_matrix, std::vector<int> &global_to_local);
  static bool ForwardEliminationMPI(InType &local_matrix, const std::vector<int> &global_to_local, size_t n,
                                    size_t cols, int rank, int size);
  static void EliminateColumnMPI(InType &local_matrix, const std::vector<int> &global_to_local, size_t k, size_t n,
                                 size_t cols, const std::vector<double> &pivot_row);
  static size_t GetGlobalIndex(const std::vector<int> &global_to_local, size_t local_idx, size_t n);
  static std::vector<double> BackSubstitutionMPI(const InType &local_matrix, const std::vector<int> &global_to_local,
                                                 size_t n, size_t cols, int rank, int size);
};

}  // namespace shilin_n_gauss_band_horizontal_scheme
