#include <gtest/gtest.h>

#include <cstddef>

#include "kolotukhin_a_merge_sort_doubles/common/include/common.hpp"
#include "kolotukhin_a_merge_sort_doubles/mpi/include/ops_mpi.hpp"
#include "kolotukhin_a_merge_sort_doubles/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kolotukhin_a_merge_sort_doubles {

class KolotukhinAMergeSortDoublesPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  std::size_t kCount_ = 1000000;
  void SetUp() override {
    test_data_.resize(kCount_);
    for (std::size_t i = 0; i < kCount_; ++i) {
      auto value = static_cast<double>(kCount_ - i);
      if (i % 3 == 0) {
        value = -value;
      }
      if (i % 7 == 0) {
        value += 0.25;
      }
      test_data_[i] = value;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::get<1>(output_data) == 0) {
      for (std::size_t i = 1; i < std::get<0>(output_data).size(); ++i) {
        if (std::get<0>(output_data)[i] < std::get<0>(output_data)[i - 1]) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return test_data_;
  }

 private:
  InType test_data_;
};

TEST_P(KolotukhinAMergeSortDoublesPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KolotukhinAMergeSortDoublesMPI, KolotukhinAMergeSortDoublesSEQ>(
        PPC_SETTINGS_kolotukhin_a_merge_sort_doubles);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KolotukhinAMergeSortDoublesPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KolotukhinAMergeSortDoublesPerfTests, kGtestValues, kPerfTestName);

}  // namespace kolotukhin_a_merge_sort_doubles
