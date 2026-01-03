#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "safronov_m_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"
#include "safronov_m_quicksort_with_batcher_even_odd_merge/mpi/include/ops_mpi.hpp"
#include "safronov_m_quicksort_with_batcher_even_odd_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace safronov_m_quicksort_with_batcher_even_odd_merge {

class SafronovMQuicksortWithBatcherEvenOddMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 70000000;
  InType input_data_;
  OutType res_;

  void SetUp() override {
    std::vector<int> vec(kCount_);
    for (int i = 0; i < kCount_; i++) {
      vec[i] = kCount_ - i;
    }
    input_data_ = vec;
    // std::sort(vec.begin(), vec.end());
    std::ranges::sort(vec);
    res_ = vec;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return res_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SafronovMQuicksortWithBatcherEvenOddMergePerfTests, QuicksortWithBatcherEvenOddMergePerf) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SafronovMQuicksortWithBatcherEvenOddMergeMPI,
                                                       SafronovMQuicksortWithBatcherEvenOddMergeSEQ>(
    PPC_SETTINGS_safronov_m_quicksort_with_batcher_even_odd_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SafronovMQuicksortWithBatcherEvenOddMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(QuicksortWithBatcherEvenOddMergePerf, SafronovMQuicksortWithBatcherEvenOddMergePerfTests,
                         kGtestValues, kPerfTestName);

}  // namespace safronov_m_quicksort_with_batcher_even_odd_merge
