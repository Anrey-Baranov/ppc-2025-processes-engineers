#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "marin_l_linear_filter_vertical/common/include/common.hpp"
#include "marin_l_linear_filter_vertical/mpi/include/ops_mpi.hpp"
#include "marin_l_linear_filter_vertical/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace marin_l_linear_filter_vertical {

namespace {

constexpr std::array<std::array<int, 3>, 3> kGaussKernel = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
constexpr int kKernelSum = 16;

std::vector<uint8_t> ApplyGaussianFilterPerf(const std::vector<uint8_t> &input, int width, int height) {
  std::vector<uint8_t> output(input.size());

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int sum = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int ny = row + ky;
          int nx = col + kx;
          uint8_t pixel_value = 0;
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            pixel_value = input[(ny * width) + nx];
          }
          sum += pixel_value * kGaussKernel.at(ky + 1).at(kx + 1);
        }
      }
      output[(row * width) + col] = static_cast<uint8_t>(std::clamp(sum / kKernelSum, 0, 255));
    }
  }
  return output;
}

}  // namespace

class MarinLLinearFilterVerticalPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kWidth = 3000;
  static constexpr int kHeight = 3000;
  InType input_data_{};
  std::vector<uint8_t> expected_output_;

  void SetUp() override {
    input_data_.width = kWidth;
    input_data_.height = kHeight;
    input_data_.pixels.resize(static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight));
    std::mt19937 gen(static_cast<unsigned int>(input_data_.pixels.size()));
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &pixel : input_data_.pixels) {
      pixel = static_cast<uint8_t>(dist(gen));
    }
    expected_output_ = ApplyGaussianFilterPerf(input_data_.pixels, kWidth, kHeight);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MarinLLinearFilterVerticalPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MarinLLinearFilterVerticalMPI, MarinLLinearFilterVerticalSEQ>(
        PPC_SETTINGS_marin_l_linear_filter_vertical);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MarinLLinearFilterVerticalPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MarinLLinearFilterVerticalPerfTests, kGtestValues, kPerfTestName);

}  // namespace marin_l_linear_filter_vertical
