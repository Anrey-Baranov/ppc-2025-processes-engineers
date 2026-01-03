#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "buzulukskiy_d_sort_batcher/common/include/common.hpp"
#include "buzulukskiy_d_sort_batcher/mpi/include/ops_mpi.hpp"
#include "buzulukskiy_d_sort_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace buzulukskiy_d_sort_batcher {

class BuzulukskiyDSortBatcherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_.resize(kCount);
    for (std::size_t idx = 0; idx < kCount; ++idx) {
      input_data_[idx] = static_cast<int>((((kCount - idx) * 7) % 10000) - 5000);
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output) final {
    std::vector<int> expected_data = input_data_;
    std::ranges::sort(expected_data);
    return output == expected_data;
  }

 private:
  static constexpr std::size_t kCount = 500000;
  InType input_data_;
};

TEST_P(BuzulukskiyDSortBatcherPerfTests, RunPerf) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks = ppc::util::MakeAllPerfTasks<InType, BuzulukskiyDSortBatcherSEQ, BuzulukskiyDSortBatcherMPI>(
    PPC_SETTINGS_buzulukskiy_d_sort_batcher);

const auto kPerfValues = ppc::util::TupleToGTestValues(kPerfTasks);

INSTANTIATE_TEST_SUITE_P(SortBatcherPerfTests, BuzulukskiyDSortBatcherPerfTests, kPerfValues);

}  // namespace buzulukskiy_d_sort_batcher
