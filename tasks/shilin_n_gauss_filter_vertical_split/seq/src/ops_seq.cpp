#include "shilin_n_gauss_filter_vertical_split/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "shilin_n_gauss_filter_vertical_split/common/include/common.hpp"

namespace shilin_n_gauss_filter_vertical_split {

ShilinNGaussFilterVerticalSplitSEQ::ShilinNGaussFilterVerticalSplitSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<uint8_t>();
}

bool ShilinNGaussFilterVerticalSplitSEQ::ValidationImpl() {
  const InType &input = GetInput();
  if (input.width <= 0 || input.height <= 0 || input.channels <= 0) {
    return false;
  }

  size_t expected_size =
      static_cast<size_t>(input.width) * static_cast<size_t>(input.height) * static_cast<size_t>(input.channels);
  return input.pixels.size() == expected_size;
}

bool ShilinNGaussFilterVerticalSplitSEQ::PreProcessingImpl() {
  const InType &input = GetInput();
  size_t output_size =
      static_cast<size_t>(input.width) * static_cast<size_t>(input.height) * static_cast<size_t>(input.channels);
  GetOutput() = std::vector<uint8_t>(output_size, 0);
  return true;
}

bool ShilinNGaussFilterVerticalSplitSEQ::RunImpl() {
  const InType &input = GetInput();
  std::vector<uint8_t> &output = GetOutput();

  ApplyGaussianKernel(input.pixels, output, input.width, input.height, input.channels);

  return true;
}

bool ShilinNGaussFilterVerticalSplitSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

double ShilinNGaussFilterVerticalSplitSEQ::GetPixelValue(const std::vector<uint8_t> &pixels, int x, int y, int width,
                                                         int height, int channels, int channel) {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return 0.0;
  }

  size_t index = ((static_cast<size_t>(y) * static_cast<size_t>(width)) * static_cast<size_t>(channels)) +
                 ((static_cast<size_t>(x) * static_cast<size_t>(channels)) + static_cast<size_t>(channel));
  return static_cast<double>(pixels[index]);
}

void ShilinNGaussFilterVerticalSplitSEQ::ApplyGaussianKernel(const std::vector<uint8_t> &input,
                                                             std::vector<uint8_t> &output, int width, int height,
                                                             int channels) {
  // ядро гаусса 3x3
  constexpr std::array<std::array<double, 3>, 3> kKernel = {{{{1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0}},
                                                             {{2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0}},
                                                             {{1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0}}}};

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      for (int ch = 0; ch < channels; ++ch) {
        double sum = 0.0;

        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            double pixel_val = GetPixelValue(input, col + kx, row + ky, width, height, channels, ch);
            const int kernel_y_idx = ky + 1;
            const int kernel_x_idx = kx + 1;
            const auto kernel_y = static_cast<size_t>(kernel_y_idx);
            const auto kernel_x = static_cast<size_t>(kernel_x_idx);
            sum += pixel_val * kKernel.at(kernel_y).at(kernel_x);
          }
        }

        size_t index = ((static_cast<size_t>(row) * static_cast<size_t>(width)) * static_cast<size_t>(channels)) +
                       ((static_cast<size_t>(col) * static_cast<size_t>(channels)) + static_cast<size_t>(ch));
        output[index] = static_cast<uint8_t>(std::clamp(sum, 0.0, 255.0));
      }
    }
  }
}

}  // namespace shilin_n_gauss_filter_vertical_split
