#if 0  // NOLINT(readability-avoid-unconditional-preprocessor-if)
// Disabled because this is a higher-numbered duplicate (task 2) and conflicts with task 1 tests.
#  include <gtest/gtest.h>

#  include <cstddef>
#  include <tuple>
#  include <vector>

#  include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/common/include/common.hpp"
#  include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/mpi/include/ops_mpi.hpp"
#  include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/seq/include/ops_seq.hpp"
#  include "util/include/perf_test_util.hpp"

namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector {

class TsibarevaERunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixRows_ = 10000;
  const int kMatrixCols_ = 10000;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    std::vector<int> flat_matrix(static_cast<size_t>(kMatrixRows_ * kMatrixCols_));
    std::vector<int> vector(kMatrixCols_);

    for (int row = 0; row < kMatrixRows_; ++row) {
      for (int col = 0; col < kMatrixCols_; ++col) {
        int value = ((row * 17) + (col * 11)) % 1000;
        if ((row + col) % 7 == 0) {
          value = -value;
        }
        flat_matrix[(row * kMatrixCols_) + col] = value;
      }
    }

    for (int col = 0; col < kMatrixCols_; ++col) {
      vector[col] = (col * 15) % 100;
      if (col % 5 == 0) {
        vector[col] = -vector[col];
      }
    }
    expected_output_.resize(kMatrixRows_, 0);
    input_data_ = std::make_tuple(flat_matrix, kMatrixRows_, kMatrixCols_, vector);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.size() == static_cast<size_t>(kMatrixRows_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TsibarevaERunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TsibarevaERibbonHorizontalMatrixMultVectorMPI,
                                                       TsibarevaERibbonHorizontalMatrixMultVectorSEQ>(
    PPC_SETTINGS_tsibareva_e_ribbon_horizontal_matrix_mult_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TsibarevaERunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TsibarevaERunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector
#endif
