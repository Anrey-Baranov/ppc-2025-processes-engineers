#include <gtest/gtest.h>

#include "makoveeva_s_simple_iteration/common/include/common.hpp"
#include "makoveeva_s_simple_iteration/mpi/include/ops_mpi.hpp"
#include "makoveeva_s_simple_iteration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace makoveeva_s_simple_iteration {

class MakoveevaSSimpleIterationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 2500;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data > 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MakoveevaSSimpleIterationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MakoveevaSSimpleIterationMPI, MakoveevaSSimpleIterationSEQ>(
        PPC_SETTINGS_makoveeva_s_simple_iteration);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MakoveevaSSimpleIterationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MakoveevaSSimpleIterationPerfTests, kGtestValues, kPerfTestName);

}  // namespace makoveeva_s_simple_iteration
