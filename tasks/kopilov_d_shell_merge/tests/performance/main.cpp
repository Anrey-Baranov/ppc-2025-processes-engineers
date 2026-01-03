#include <gtest/gtest.h>

#include <cstddef>

#include "kopilov_d_shell_merge/common/include/common.hpp"
#include "kopilov_d_shell_merge/mpi/include/ops_mpi.hpp"
#include "kopilov_d_shell_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kopilov_d_shell_merge {

namespace {

void ShellSort(InType &vec) {
  const std::size_t n = vec.size();
  for (std::size_t gap = n / 2; gap > 0; gap /= 2) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = vec[i];
      std::size_t j = i;
      while (j >= gap && vec[j - gap] > tmp) {
        vec[j] = vec[j - gap];
        j -= gap;
      }
      vec[j] = tmp;
    }
  }
}

}  // namespace

class KopilovDShellMergeRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    constexpr std::size_t kSize = 200000;
    input_data_.resize(kSize);

    unsigned int x = 17;
    for (std::size_t i = 0; i < kSize; ++i) {
      x = (x * 1103515245) + 12345;
      input_data_[i] = static_cast<int>(x);
    }

    expected_ = input_data_;
    ShellSort(expected_);
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

TEST_P(KopilovDShellMergeRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KopilovDShellMergeMPI, KopilovDShellMergeSEQ>(
    PPC_SETTINGS_kopilov_d_shell_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = KopilovDShellMergeRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KopilovDShellMergeRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kopilov_d_shell_merge
