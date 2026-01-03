#pragma once

#include <cstddef>
#include <vector>

#include "shilin_n_gauss_band_horizontal_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shilin_n_gauss_band_horizontal_scheme {

class ShilinNGaussBandHorizontalSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShilinNGaussBandHorizontalSchemeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static bool ForwardElimination(InType &augmented_matrix, size_t n, size_t cols);
  static size_t FindPivotRow(const InType &augmented_matrix, size_t k, size_t n);
  static void EliminateColumn(InType &augmented_matrix, size_t k, size_t n, size_t cols);
  static std::vector<double> BackSubstitution(const InType &augmented_matrix, size_t n, size_t cols);
};

}  // namespace shilin_n_gauss_band_horizontal_scheme
