#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "shilin_n_gauss_filter_vertical_split/common/include/common.hpp"
#include "shilin_n_gauss_filter_vertical_split/mpi/include/ops_mpi.hpp"
#include "shilin_n_gauss_filter_vertical_split/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shilin_n_gauss_filter_vertical_split {

class ShilinNGaussFilterVerticalSplitPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const int width = 500;
    const int height = 500;
    const int channels = 3;

    input_data_.width = width;
    input_data_.height = height;
    input_data_.channels = channels;

    size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
    input_data_.pixels = std::vector<uint8_t>(pixel_count);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    for (size_t i = 0; i < pixel_count; ++i) {
      input_data_.pixels[i] = static_cast<uint8_t>(dist(gen));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    size_t expected_size = static_cast<size_t>(input_data_.width) * static_cast<size_t>(input_data_.height) *
                           static_cast<size_t>(input_data_.channels);
    return output_data.size() == expected_size;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ShilinNGaussFilterVerticalSplitPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShilinNGaussFilterVerticalSplitMPI, ShilinNGaussFilterVerticalSplitSEQ>(
        PPC_SETTINGS_shilin_n_gauss_filter_vertical_split);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShilinNGaussFilterVerticalSplitPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShilinNGaussFilterVerticalSplitPerfTests, kGtestValues, kPerfTestName);

}  // namespace shilin_n_gauss_filter_vertical_split
