#include <gtest/gtest.h>

#include <cstddef>

#include "balchunayte_z_shell_batcher/common/include/common.hpp"
#include "balchunayte_z_shell_batcher/mpi/include/ops_mpi.hpp"
#include "balchunayte_z_shell_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace balchunayte_z_shell_batcher {

namespace {

void ShellSort(InType *vec) {
  auto &a = *vec;
  const std::size_t n = a.size();
  for (std::size_t gap = n / 2; gap > 0; gap /= 2) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = a[i];
      std::size_t j = i;
      while (j >= gap && a[j - gap] > tmp) {
        a[j] = a[j - gap];
        j -= gap;
      }
      a[j] = tmp;
    }
  }
}

}  // namespace

class ShellBatcherRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    constexpr std::size_t kSize = 200000;
    input_data_.resize(kSize);

    int x = 17;
    for (std::size_t i = 0; i < kSize; ++i) {
      x = (x * 1103515245) + 12345;
      input_data_[i] = x;
    }

    expected_ = input_data_;
    ShellSort(&expected_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

TEST_P(ShellBatcherRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BalchunayteZShellBatcherMPI, BalchunayteZShellBatcherSEQ>(
        PPC_SETTINGS_balchunayte_z_shell_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = ShellBatcherRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShellBatcherRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace balchunayte_z_shell_batcher
