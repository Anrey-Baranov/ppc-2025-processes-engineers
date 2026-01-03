#include <gtest/gtest.h>

#include <cmath>

#include "karpich_i_integrals_multistep_rectangle/common/include/common.hpp"
#include "karpich_i_integrals_multistep_rectangle/mpi/include/ops_mpi.hpp"
#include "karpich_i_integrals_multistep_rectangle/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace karpich_i_integrals_multistep_rectangle {

static double ComputeExpectedResult(int n) {
  double result = 0.0;
  for (int step = 1; step <= 3; step++) {
    int divisions = n * step;
    double h = 1.0 / divisions;
    double step_result = 0.0;

    for (int i = 0; i < divisions; i++) {
      for (int j = 0; j < divisions; j++) {
        for (int k = 0; k < divisions; k++) {
          double x = (i + 0.5) * h;
          double y = (j + 0.5) * h;
          double z = (k + 0.5) * h;
          step_result += (x * y * z + x * x + y * y + z * z) * h * h * h;
        }
      }
    }
    result += step_result;
  }
  return result / 3.0;
}

class KarpichIIntegralsMultistepRectanglePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 120;
  InType input_data_{};
  OutType expected_result_{};

  void SetUp() override {
    input_data_ = kCount_;
    expected_result_ = static_cast<OutType>(std::round(ComputeExpectedResult(input_data_)));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(static_cast<int>(output_data) - static_cast<int>(expected_result_)) <= 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KarpichIIntegralsMultistepRectanglePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KarpichIIntegralsMultistepRectangleMPI, KarpichIIntegralsMultistepRectangleSEQ>(
        PPC_SETTINGS_karpich_i_integrals_multistep_rectangle);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KarpichIIntegralsMultistepRectanglePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KarpichIIntegralsMultistepRectanglePerfTests, kGtestValues, kPerfTestName);

}  // namespace karpich_i_integrals_multistep_rectangle
