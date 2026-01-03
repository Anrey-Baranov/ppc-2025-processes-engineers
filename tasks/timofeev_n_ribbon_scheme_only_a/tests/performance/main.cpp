#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "timofeev_n_ribbon_scheme_only_a/common/include/common.hpp"
#include "timofeev_n_ribbon_scheme_only_a/mpi/include/ops_mpi.hpp"
#include "timofeev_n_ribbon_scheme_only_a/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace timofeev_n_ribbon_scheme_only_a {

class RibbonRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 1000;
  OutType expected_data_;
  InType input_data_;

  void SetUp() override {
    MatrixType a(kCount_, std::vector<int>(kCount_, 0));
    MatrixType b(kCount_, std::vector<int>(kCount_, 1));
    for (int i = 0; i < kCount_; i++) {
      a[i][i] = 1;
    }
    input_data_ = std::make_pair(a, b);
    expected_data_ = b;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_data_.size() || output_data[0].size() != expected_data_[0].size()) {
      return false;
    }
    bool yesyes = true;
    for (int i = 0; i < kCount_; i++) {
      for (int j = 0; j < kCount_; j++) {
        yesyes &= output_data[i][j] == expected_data_[i][j];
      }
    }
    return yesyes;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RibbonRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TimofeevNRibbonSchemeOnlyAMPI, TimofeevNRibbonSchemeOnlyASEQ>(
        PPC_SETTINGS_timofeev_n_ribbon_scheme_only_a);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RibbonRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RibbonRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace timofeev_n_ribbon_scheme_only_a
