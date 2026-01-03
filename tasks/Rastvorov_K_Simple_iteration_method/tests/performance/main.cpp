#include <gtest/gtest.h>

#include "Rastvorov_K_Simple_iteration_method/common/include/common.hpp"
#include "Rastvorov_K_Simple_iteration_method/mpi/include/ops_mpi.hpp"
#include "Rastvorov_K_Simple_iteration_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rastvorov_k_simple_iteration_method {

class RastvorovKSimpleIterationPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    input_data_ = 10000;
  }

  bool CheckTestOutputData(OutType & /*output_data*/) final {
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RastvorovKSimpleIterationPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RastvorovKSimpleIterationMethodMPI, RastvorovKSimpleIterationMethodSEQ>(
        PPC_SETTINGS_Rastvorov_K_Simple_iteration_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunModeTests, RastvorovKSimpleIterationPerfTestProcesses, kGtestValues,
                         RastvorovKSimpleIterationPerfTestProcesses::CustomPerfTestName);

}  // namespace rastvorov_k_simple_iteration_method
