#include "kondrashova_v_gauss_filter_vertical_split/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

const std::array<std::array<int, 3>, 3> KondrashovaVGaussFilterVerticalSplitSEQ::kGaussKernel = {
    {{{1, 2, 1}}, {{2, 4, 2}}, {{1, 2, 1}}}};
const int KondrashovaVGaussFilterVerticalSplitSEQ::kGaussKernelSum = 16;

uint8_t KondrashovaVGaussFilterVerticalSplitSEQ::ApplyGaussToPixel(const std::vector<uint8_t> &pixels, int width,
                                                                   int height, int channels, int px, int py,
                                                                   int channel) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = std::clamp(px + kx, 0, width - 1);
      int ny = std::clamp(py + ky, 0, height - 1);

      int idx = (((ny * width) + nx) * channels) + channel;
      auto kernel_row = static_cast<size_t>(ky) + 1;
      auto kernel_col = static_cast<size_t>(kx) + 1;
      sum += pixels[idx] * kGaussKernel.at(kernel_row).at(kernel_col);
    }
  }

  return static_cast<uint8_t>(std::clamp(sum / kGaussKernelSum, 0, 255));
}

KondrashovaVGaussFilterVerticalSplitSEQ::KondrashovaVGaussFilterVerticalSplitSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KondrashovaVGaussFilterVerticalSplitSEQ::ValidationImpl() {
  const auto &input = GetInput();

  auto expected_size = static_cast<size_t>(input.width) * input.height * input.channels;
  return input.pixels.size() == expected_size && input.width >= 3 && input.height >= 3 && input.channels >= 1 &&
         input.channels <= 4;
}

bool KondrashovaVGaussFilterVerticalSplitSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  output.width = input.width;
  output.height = input.height;
  output.channels = input.channels;
  output.pixels.resize(input.pixels.size());

  return true;
}

bool KondrashovaVGaussFilterVerticalSplitSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  int width = input.width;
  int height = input.height;
  int channels = input.channels;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      for (int ch = 0; ch < channels; ++ch) {
        int idx = (((row * width) + col) * channels) + ch;
        output.pixels[idx] = ApplyGaussToPixel(input.pixels, width, height, channels, col, row, ch);
      }
    }
  }

  return true;
}

bool KondrashovaVGaussFilterVerticalSplitSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kondrashova_v_gauss_filter_vertical_split
