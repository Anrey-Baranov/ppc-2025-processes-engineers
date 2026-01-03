#include <gtest/gtest.h>

#include <string>

#include "Rastvorov_K_Number_of_character_alternations/common/include/common.hpp"
#include "Rastvorov_K_Number_of_character_alternations/mpi/include/ops_mpi.hpp"
#include "Rastvorov_K_Number_of_character_alternations/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rastvorov_k_number_of_character_alternations {

class RastvorovKNumberAfCharacterAlternationsRunPerfTestProcesses
    : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    input_data_ = 10000000;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RastvorovKNumberAfCharacterAlternationsRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RastvorovKNumberAfCharacterAlternationsMPI,
                                RastvorovKNumberAfCharacterAlternationsSEQ>(PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

namespace {

using ParamType = RastvorovKNumberAfCharacterAlternationsRunPerfTestProcesses::ParamType;

::testing::internal::ParamGenerator<ParamType> RastvorovPerfEvalGenerator() {
  return kGtestValues;
}

std::string RastvorovPerfEvalGenerateName(const ::testing::TestParamInfo<ParamType> &info) {
  return RastvorovKNumberAfCharacterAlternationsRunPerfTestProcesses::CustomPerfTestName(info);
}

const int kRastvorovPerfDummy =
    ::testing::UnitTest::GetInstance()
        ->parameterized_test_registry()
        .GetTestSuitePatternHolder<RastvorovKNumberAfCharacterAlternationsRunPerfTestProcesses>(
            "RastvorovKNumberAfCharacterAlternationsRunPerfTestProcesses",
            ::testing::internal::CodeLocation(__FILE__, __LINE__))
        ->AddTestSuiteInstantiation("RunModeTests", &RastvorovPerfEvalGenerator, &RastvorovPerfEvalGenerateName,
                                    __FILE__, __LINE__);

}  // namespace

}  // namespace rastvorov_k_number_of_character_alternations
