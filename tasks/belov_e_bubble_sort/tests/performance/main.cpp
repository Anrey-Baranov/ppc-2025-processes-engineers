#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "belov_e_bubble_sort/common/include/common.hpp"
#include "belov_e_bubble_sort/mpi/include/ops_mpi.hpp"
#include "belov_e_bubble_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace belov_e_bubble_sort {
class BelovEBubbleSortRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t size_params_ = 20000;
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

TEST_P(BelovEBubbleSortRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BelovEBubbleSortMPI, BelovEBubbleSortSEQ>(PPC_SETTINGS_belov_e_bubble_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BelovEBubbleSortRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BelovEBubbleSortRunPerfTestsProcesses, kGtestValues, kPerfTestName);
}  // namespace belov_e_bubble_sort
