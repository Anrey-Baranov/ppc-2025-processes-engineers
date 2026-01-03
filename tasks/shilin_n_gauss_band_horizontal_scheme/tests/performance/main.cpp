#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "shilin_n_gauss_band_horizontal_scheme/common/include/common.hpp"
#include "shilin_n_gauss_band_horizontal_scheme/mpi/include/ops_mpi.hpp"
#include "shilin_n_gauss_band_horizontal_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shilin_n_gauss_band_horizontal_scheme {

class ShilinNGaussBandHorizontalSchemePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const int matrix_size = 1000;
    const int band_width = 5;

    input_data_ = InType(static_cast<size_t>(matrix_size));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.1, 10.0);

    for (int i = 0; i < matrix_size; ++i) {
      input_data_[static_cast<size_t>(i)] = std::vector<double>(static_cast<size_t>(matrix_size) + 1, 0.0);

      for (int j = 0; j < matrix_size; ++j) {
        if (std::abs(i - j) <= band_width) {
          input_data_[static_cast<size_t>(i)][static_cast<size_t>(j)] = dist(gen);
        }
      }

      double sum = 0.0;
      for (int j = 0; j < matrix_size; ++j) {
        if (i != j) {
          sum += std::abs(input_data_[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        }
      }
      input_data_[static_cast<size_t>(i)][static_cast<size_t>(i)] = sum + 1.0;
      input_data_[static_cast<size_t>(i)][static_cast<size_t>(matrix_size)] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.size() == 1000;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ShilinNGaussBandHorizontalSchemePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShilinNGaussBandHorizontalSchemeMPI, ShilinNGaussBandHorizontalSchemeSEQ>(
        PPC_SETTINGS_shilin_n_gauss_band_horizontal_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShilinNGaussBandHorizontalSchemePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShilinNGaussBandHorizontalSchemePerfTests, kGtestValues, kPerfTestName);

}  // namespace shilin_n_gauss_band_horizontal_scheme
