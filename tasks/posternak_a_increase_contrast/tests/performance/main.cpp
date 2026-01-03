#include <gtest/gtest.h>

#include <cstddef>

#include "posternak_a_increase_contrast/common/include/common.hpp"
#include "posternak_a_increase_contrast/mpi/include/ops_mpi.hpp"
#include "posternak_a_increase_contrast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace posternak_a_increase_contrast {

class PosternakAIncreaseContrastPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t kPixelsCount_ = static_cast<size_t>(8192) * 8192;  // 8к изображение
  InType input_data_;

  void SetUp() override {
    input_data_.resize(kPixelsCount_);
    for (size_t i = 0; i < kPixelsCount_; i++) {
      // повторяющийся блок от 100 до 150
      input_data_[i] = static_cast<unsigned char>(100 + (i % 51));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // считать значения - безумие, поэтому проверяем осмысленность результата
    return !output_data.empty() && output_data.size() == kPixelsCount_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PosternakAIncreaseContrastPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PosternakAIncreaseContrastMPI, PosternakAIncreaseContrastSEQ>(
        PPC_SETTINGS_posternak_a_increase_contrast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PosternakAIncreaseContrastPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PosternakAIncreaseContrastPerfTests, kGtestValues, kPerfTestName);

}  // namespace posternak_a_increase_contrast
