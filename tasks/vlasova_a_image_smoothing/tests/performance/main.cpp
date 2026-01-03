#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "vlasova_a_image_smoothing/common/include/common.hpp"
#include "vlasova_a_image_smoothing/mpi/include/ops_mpi.hpp"
#include "vlasova_a_image_smoothing/seq/include/ops_seq.hpp"

namespace vlasova_a_image_smoothing {

class VlasovaARunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int width = 512;
    const int height = 512;

    input_data_.width = width;
    input_data_.height = height;
    input_data_.data.resize(static_cast<std::size_t>(width) * height);

    for (std::size_t idx = 0; idx < input_data_.data.size(); ++idx) {
      input_data_.data[idx] = static_cast<std::uint8_t>((idx * 37) % 256);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.width == input_data_.width && output_data.height == input_data_.height &&
           !output_data.data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(VlasovaARunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VlasovaAImageSmoothingMPI, VlasovaAImageSmoothingSEQ>(
    PPC_SETTINGS_vlasova_a_image_smoothing);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VlasovaARunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerformanceTests, VlasovaARunPerfTests, kGtestValues, kPerfTestName);

}  // namespace vlasova_a_image_smoothing
