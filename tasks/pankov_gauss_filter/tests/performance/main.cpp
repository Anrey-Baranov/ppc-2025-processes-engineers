#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pankov_gauss_filter/common/include/common.hpp"
#include "pankov_gauss_filter/mpi/include/ops_mpi.hpp"
#include "pankov_gauss_filter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pankov_gauss_filter {

namespace {

inline int ClampInt(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

inline std::uint8_t ClampToByte(int v) {
  v = std::max(0, std::min(v, 255));
  return static_cast<std::uint8_t>(v);
}

constexpr std::array<std::array<int, 3>, 3> kGaussianKernel3x3 = {{{{1, 2, 1}}, {{2, 4, 2}}, {{1, 2, 1}}}};
constexpr int kGaussianDiv = 16;

std::uint8_t ConvolveGaussian3x3Clamp(const Image &image, int row, int col, int channel) {
  const int width = image.width;
  const int height = image.height;
  const int channels = image.channels;

  int acc = 0;
  for (std::size_t kernel_row = 0; kernel_row < 3; ++kernel_row) {
    const int d_row = static_cast<int>(kernel_row) - 1;
    const int src_row = ClampInt(row + d_row, 0, height - 1);
    for (std::size_t kernel_col = 0; kernel_col < 3; ++kernel_col) {
      const int d_col = static_cast<int>(kernel_col) - 1;
      const int src_col = ClampInt(col + d_col, 0, width - 1);
      const int weight = kGaussianKernel3x3.at(kernel_row).at(kernel_col);
      const std::size_t idx =
          ((static_cast<std::size_t>(src_row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(src_col)) *
           static_cast<std::size_t>(channels)) +
          static_cast<std::size_t>(channel);
      acc += weight * static_cast<int>(image.data[idx]);
    }
  }

  const int rounded = (acc + (kGaussianDiv / 2)) / kGaussianDiv;
  return ClampToByte(rounded);
}

Image ApplyGaussian3x3Clamp(const Image &in) {
  Image out;
  out.width = in.width;
  out.height = in.height;
  out.channels = in.channels;
  out.data.assign(in.data.size(), 0);
  if (in.width == 0 || in.height == 0) {
    return out;
  }

  const int width = in.width;
  const int height = in.height;
  const int channels = in.channels;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      for (int channel = 0; channel < channels; ++channel) {
        const std::size_t out_idx =
            ((static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(col)) *
             static_cast<std::size_t>(channels)) +
            static_cast<std::size_t>(channel);
        out.data[out_idx] = ConvolveGaussian3x3Clamp(in, row, col, channel);
      }
    }
  }
  return out;
}

}  // namespace

class PankovGaussFilterRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    const int w = 4096;
    const int h = 4096;
    const int ch = 1;
    input_data_.width = w;
    input_data_.height = h;
    input_data_.channels = ch;
    input_data_.data.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(ch));

    for (int row = 0; row < h; ++row) {
      for (int col = 0; col < w; ++col) {
        const std::size_t idx =
            (static_cast<std::size_t>(row) * static_cast<std::size_t>(w)) + static_cast<std::size_t>(col);
        input_data_.data[idx] = static_cast<std::uint8_t>((col + row) % 256);
      }
    }

    expected_output_ = ApplyGaussian3x3Clamp(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.width == expected_output_.width && output_data.height == expected_output_.height &&
           output_data.channels == expected_output_.channels && output_data.data == expected_output_.data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PankovGaussFilterRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PankovGaussFilterMPI, PankovGaussFilterSEQ>(PPC_SETTINGS_pankov_gauss_filter);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = PankovGaussFilterRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PankovGaussFilterRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace pankov_gauss_filter
