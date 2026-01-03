#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "pankov_gauss_filter/common/include/common.hpp"
#include "pankov_gauss_filter/mpi/include/ops_mpi.hpp"
#include "pankov_gauss_filter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace pankov_gauss_filter {

using LocalTestType = InType;

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

Image MakeImage(int w, int h, int ch, const std::vector<std::uint8_t> &data) {
  return Image{w, h, ch, data};
}

}  // namespace

class PankovGaussFilterRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, LocalTestType> {
 public:
  static std::string PrintTestParam(const LocalTestType &test_param) {
    return "img_" + std::to_string(test_param.width) + "x" + std::to_string(test_param.height) + "_ch" +
           std::to_string(test_param.channels);
  }

 protected:
  void SetUp() override {
    LocalTestType input = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = input;
    expected_output_ = ApplyGaussian3x3Clamp(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.width == expected_output_.width && output_data.height == expected_output_.height &&
           output_data.channels == expected_output_.channels && output_data.data == expected_output_.data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(PankovGaussFilterRunFuncTestsProcesses, Gaussian3x3Filter) {
  ExecuteTest(GetParam());
}

const std::array<LocalTestType, 5> kTestParam = {
    MakeImage(1, 1, 1, {128}),
    MakeImage(2, 2, 1, {0, 64, 128, 255}),
    MakeImage(3, 3, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8}),
    MakeImage(4, 2, 3,
              {0, 0, 0, 10, 20, 30, 40, 50, 60, 255, 128, 64, 5, 10, 15, 25, 35, 45, 60, 70, 80, 90, 100, 110}),
    MakeImage(5, 4, 1, {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190}),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<PankovGaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_pankov_gauss_filter),
                   ppc::util::AddFuncTask<PankovGaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_pankov_gauss_filter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kFuncTestName =
    PankovGaussFilterRunFuncTestsProcesses::PrintFuncTestName<PankovGaussFilterRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussFilterTests, PankovGaussFilterRunFuncTestsProcesses, kGtestValues, kFuncTestName);

}  // namespace pankov_gauss_filter
