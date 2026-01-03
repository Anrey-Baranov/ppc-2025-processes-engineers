#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"
#include "tsibareva_e_matrix_column_max/mpi/include/ops_mpi.hpp"
#include "tsibareva_e_matrix_column_max/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tsibareva_e_matrix_column_max {

class TsibarevaERunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixRows_ = 10000;
  const int kMatrixCols_ = 10000;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_.resize(kMatrixRows_, std::vector<int>(kMatrixCols_));
    expected_output_.resize(kMatrixCols_, std::numeric_limits<int>::min());

    int row_middle = kMatrixRows_ / 2;

    for (int j = 0; j < kMatrixCols_; ++j) {
      for (int i = 0; i < kMatrixRows_; ++i) {
        int generate_value = 0;
        if (i == row_middle) {
          generate_value = 1000000 + j;
        } else {
          generate_value = ((i * kMatrixCols_) + j) % 1000;
        }

        input_data_[i][j] = generate_value;

        expected_output_[j] = std::max(generate_value, expected_output_[j]);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TsibarevaERunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TsibarevaEMatrixColumnMaxMPI, TsibarevaEMatrixColumnMaxSEQ>(
        PPC_SETTINGS_tsibareva_e_matrix_column_max);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TsibarevaERunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TsibarevaERunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace tsibareva_e_matrix_column_max
