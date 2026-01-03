#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>

#include "dorogin_v_bitwise_simple_merge/common/include/common.hpp"
#include "dorogin_v_bitwise_simple_merge/mpi/include/ops_mpi.hpp"
#include "dorogin_v_bitwise_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dorogin_v_bitwise_simple_merge {

class DoroginVBitwiseSimpleMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 1500000;
  InType input_data_;

  void SetUp() override {
    input_data_.resize(kCount_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);
    for (int i = 0; i < kCount_; i++) {
      input_data_[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(kCount_)) {
      return false;
    }
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DoroginVBitwiseSimpleMergePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DoroginVBitwiseSimpleMergeMPI, DoroginVBitwiseSimpleMergeSEQ>(
        PPC_SETTINGS_dorogin_v_bitwise_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DoroginVBitwiseSimpleMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DoroginVBitwiseSimpleMergePerfTests, kGtestValues, kPerfTestName);

}  // namespace dorogin_v_bitwise_simple_merge
