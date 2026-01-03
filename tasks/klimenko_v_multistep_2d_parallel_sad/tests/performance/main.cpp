#include <gtest/gtest.h>

#include <cmath>

#include "klimenko_v_multistep_2d_parallel_sad/common/include/common.hpp"
#include "klimenko_v_multistep_2d_parallel_sad/mpi/include/ops_mpi.hpp"
#include "klimenko_v_multistep_2d_parallel_sad/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace klimenko_v_multistep_2d_parallel_sad {

namespace {

double SpherePerfFunc(double x, double y) {
  double sum = 0.0;
  for (int i = 0; i < 1000; ++i) {
    sum += std::sin(x * i) * std::cos(y * i);
  }
  return sum;
}

}  // namespace

class KlimenkoV2DParallelSadPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kMaxIterations = 2000;
  InType input_data;

  void SetUp() override {
    input_data.func = SpherePerfFunc;
    input_data.x_min = -50.0;
    input_data.x_max = 50.0;
    input_data.y_min = -50.0;
    input_data.y_max = 50.0;
    input_data.epsilon = 1e-6;
    input_data.r_param = 2.5;
    input_data.max_iterations = kMaxIterations;
  }

  bool CheckTestOutputData(OutType &output) final {
    return output.iterations >= 0 && output.x_opt >= input_data.x_min && output.x_opt <= input_data.x_max &&
           output.y_opt >= input_data.y_min && output.y_opt <= input_data.y_max;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(KlimenkoV2DParallelSadPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KlimenkoV2DParallelSadMPI, KlimenkoV2DParallelSadSEQ>(
    PPC_SETTINGS_klimenko_v_multistep_2d_parallel_sad);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KlimenkoV2DParallelSadPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KlimenkoV2DParallelSadPerfTests, kGtestValues, kPerfTestName);

}  // namespace klimenko_v_multistep_2d_parallel_sad
