#include <gtest/gtest.h>

#include <vector>

#include "timofeev_n_radix_merge_sort/common/include/common.hpp"
#include "timofeev_n_radix_merge_sort/mpi/include/ops_mpi.hpp"
#include "timofeev_n_radix_merge_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace timofeev_n_radix_merge_sort {

class RadixMergeRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000000;
  OutType expected_data_;
  InType input_data_;

  void SetUp() override {
    std::vector<int> a(kCount_);
    std::vector<int> b(kCount_);
    for (int i = 0; i < kCount_; i++) {
      a[i] = kCount_ - i;
    }
    for (int i = 0; i < kCount_; i++) {
      b[i] = i + 1;
    }
    input_data_ = a;
    expected_data_ = b;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_data_.size()) {
      return false;
    }
    bool yesyes = true;
    for (int i = 0; i < kCount_; i++) {
      yesyes &= output_data[i] == expected_data_[i];
    }
    return yesyes;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RadixMergeRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TimofeevNRadixMergeMPI, TimofeevNRadixMergeSEQ>(
    PPC_SETTINGS_timofeev_n_radix_merge_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RadixMergeRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RadixMergeRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace timofeev_n_radix_merge_sort
