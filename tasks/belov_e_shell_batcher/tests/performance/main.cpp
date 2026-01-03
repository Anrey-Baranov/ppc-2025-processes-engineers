#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "belov_e_shell_batcher/common/include/common.hpp"
#include "belov_e_shell_batcher/mpi/include/ops_mpi.hpp"
#include "belov_e_shell_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace belov_e_shell_batcher {
class BelovEShellBatcherRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t size_params_ = 3000000;
  InType input_data_;

  void SetUp() override {
    input_data_.resize(size_params_);
    for (size_t i = 0; i < size_params_; i++) {
      input_data_[i] = static_cast<int>(size_params_) - static_cast<int>(i);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BelovEShellBatcherRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BelovEShellBatcherMPI, BelovEShellBatcherSEQ>(
    PPC_SETTINGS_belov_e_shell_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BelovEShellBatcherRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BelovEShellBatcherRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace belov_e_shell_batcher
