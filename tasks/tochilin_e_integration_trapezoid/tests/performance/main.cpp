#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include "tochilin_e_integration_trapezoid/common/include/common.hpp"
#include "tochilin_e_integration_trapezoid/mpi/include/ops_mpi.hpp"
#include "tochilin_e_integration_trapezoid/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tochilin_e_integration_trapezoid {

class TochilinEIntegrationTrapezoidPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kNumIntervals_ = 10000000;
  InType input_data_{};
  double expected_result_ = 0.0;

  void SetUp() override {
    input_data_.lower_bound = 0.0;
    input_data_.upper_bound = std::numbers::pi;
    input_data_.num_intervals = kNumIntervals_;
    input_data_.function = [](double x) { return std::sin(x); };
    expected_result_ = 2.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double tolerance = 1e-4;
    return std::abs(output_data - expected_result_) < tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TochilinEIntegrationTrapezoidPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TochilinEIntegrationTrapezoidMPI, TochilinEIntegrationTrapezoidSEQ>(
        PPC_SETTINGS_tochilin_e_integration_trapezoid);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TochilinEIntegrationTrapezoidPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TochilinEIntegrationTrapezoidPerfTests, kGtestValues, kPerfTestName);

}  // namespace tochilin_e_integration_trapezoid
