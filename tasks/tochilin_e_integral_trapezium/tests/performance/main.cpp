#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "tochilin_e_integral_trapezium/common/include/common.hpp"
#include "tochilin_e_integral_trapezium/mpi/include/ops_mpi.hpp"
#include "tochilin_e_integral_trapezium/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tochilin_e_integral_trapezium {

namespace {

double PerfFunc(const std::vector<double> &point) {
  double sum = 0.0;
  for (std::size_t idx = 0; idx < point.size(); ++idx) {
    sum += (std::sin(point[idx]) * std::cos(point[idx])) + (point[idx] * point[idx]);
  }
  return sum;
}

}  // namespace

class TochilinEIntegralTrapeziumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    input_data_.lower_bounds = {0.0, 0.0, 0.0};
    input_data_.upper_bounds = {1.0, 1.0, 1.0};
    input_data_.num_steps = 100;
    input_data_.func = PerfFunc;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data > 0.0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TochilinEIntegralTrapeziumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TochilinEIntegralTrapeziumMPI, TochilinEIntegralTrapeziumSEQ>(
        PPC_SETTINGS_tochilin_e_integral_trapezium);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TochilinEIntegralTrapeziumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TochilinEIntegralTrapeziumPerfTests, kGtestValues, kPerfTestName);

}  // namespace tochilin_e_integral_trapezium
