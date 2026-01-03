#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "dorogin_v_contrast_enhancement/common/include/common.hpp"
#include "dorogin_v_contrast_enhancement/mpi/include/ops_mpi.hpp"
#include "dorogin_v_contrast_enhancement/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dorogin_v_contrast_enhancement {

class DoroginVContrastEnhancementPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kSize = 25000000;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_.resize(kSize);
    for (int i = 0; i < kSize; ++i) {
      input_data_[i] = static_cast<uint8_t>(i % 256);
    }

    expected_output_.resize(kSize);
    for (int i = 0; i < kSize; ++i) {
      expected_output_[i] = static_cast<uint8_t>(i % 256);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DoroginVContrastEnhancementPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DoroginVContrastEnhancementMPI, DoroginVContrastEnhancementSEQ>(
        PPC_SETTINGS_dorogin_v_contrast_enhancement);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DoroginVContrastEnhancementPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DoroginVContrastEnhancementPerfTests, kGtestValues, kPerfTestName);

}  // namespace dorogin_v_contrast_enhancement
