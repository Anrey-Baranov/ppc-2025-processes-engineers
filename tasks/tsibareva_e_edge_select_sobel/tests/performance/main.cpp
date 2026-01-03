#if 0  // NOLINT(readability-avoid-unconditional-preprocessor-if)
// Disabled because this is a higher-numbered duplicate (task 3) and conflicts with task 1 tests.
#  include <gtest/gtest.h>

#  include <algorithm>
#  include <tuple>
#  include <vector>

#  include "tsibareva_e_edge_select_sobel/common/include/common.hpp"
#  include "tsibareva_e_edge_select_sobel/mpi/include/ops_mpi.hpp"
#  include "tsibareva_e_edge_select_sobel/seq/include/ops_seq.hpp"
#  include "util/include/perf_test_util.hpp"

namespace tsibareva_e_edge_select_sobel {

class TsibarevaERunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    int height = 3000;
    int width = 3000;

    std::vector<std::vector<int>> image_data(height, std::vector<int>(width));

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int val = ((i * 90) + (j * 111)) % 109;
        image_data[i][j] = val;
      }
    }

    for (int i = 0; i < height; ++i) {
      image_data[i][width / 2] = 255;
      if (((width / 2) + 1) < width) {
        image_data[i][(width / 2) + 1] = 255;
      }
    }

    for (int j = 0; j < width; ++j) {
      image_data[height / 3][j] = 255;
      if (((height / 3) + 1) < height) {
        image_data[(height / 3) + 1][j] = 255;
      }
    }

    for (int k = 0; k < std::min(height, width); ++k) {
      image_data[k][k] = 255;
      if ((k + 1) < width) {
        image_data[k][k + 1] = 255;
      }
    }

    std::vector<int> flat_data;
    for (const auto &row : image_data) {
      flat_data.insert(flat_data.end(), row.begin(), row.end());
    }

    int threshold = 100;
    input_data_ = std::make_tuple(flat_data, height, width, threshold);
    expected_output_ = std::vector<int>(flat_data.size(), 0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data.size() == expected_output_.size());
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TsibarevaERunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TsibarevaEEdgeSelectSobelMPI, TsibarevaEEdgeSelectSobelSEQ>(
        PPC_SETTINGS_tsibareva_e_edge_select_sobel);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TsibarevaERunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TsibarevaERunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace tsibareva_e_edge_select_sobel
#endif
