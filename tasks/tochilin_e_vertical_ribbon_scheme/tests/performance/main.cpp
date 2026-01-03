#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "tochilin_e_vertical_ribbon_scheme/common/include/common.hpp"
#include "tochilin_e_vertical_ribbon_scheme/mpi/include/ops_mpi.hpp"
#include "tochilin_e_vertical_ribbon_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tochilin_e_vertical_ribbon_scheme {

class TochilinEVerticalRibbonSchemePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kSize = 8000;
  InType input_data_{};
  std::vector<double> expected_;

  void SetUp() override {
    input_data_.rows = kSize;
    input_data_.cols = kSize;
    input_data_.matrix.resize(static_cast<std::size_t>(kSize) * kSize);
    input_data_.vector.resize(kSize);
    expected_.resize(kSize, 0.0);

    for (int j = 0; j < kSize; j++) {
      input_data_.vector[j] = static_cast<double>((j % 10) + 1);
      for (int i = 0; i < kSize; i++) {
        input_data_.matrix[static_cast<std::size_t>(j * kSize) + i] = static_cast<double>(((i + j) % 20) - 10);
      }
    }

    for (int j = 0; j < kSize; j++) {
      for (int i = 0; i < kSize; i++) {
        expected_[i] += input_data_.matrix[static_cast<std::size_t>(j * kSize) + i] * input_data_.vector[j];
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < expected_.size(); i++) {
      if (std::abs(output_data[i] - expected_[i]) > 1e-6) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TochilinEVerticalRibbonSchemePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TochilinEVerticalRibbonSchemeMPI, TochilinEVerticalRibbonSchemeSEQ>(
        PPC_SETTINGS_tochilin_e_vertical_ribbon_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TochilinEVerticalRibbonSchemePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TochilinEVerticalRibbonSchemePerfTests, kGtestValues, kPerfTestName);

}  // namespace tochilin_e_vertical_ribbon_scheme
