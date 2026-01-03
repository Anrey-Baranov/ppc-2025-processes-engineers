#include "marin_l_linear_filter_vertical/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "marin_l_linear_filter_vertical/common/include/common.hpp"

namespace marin_l_linear_filter_vertical {

namespace {
constexpr std::array<std::array<int, 3>, 3> kGaussKernel = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
constexpr int kKernelSum = 16;

uint8_t ApplyKernel(const std::vector<uint8_t> &pixels, int width, int height, int row, int col) {
  int sum = 0;
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int ny = row + ky;
      int nx = col + kx;
      uint8_t pixel_value = 0;
      if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
        pixel_value = pixels[(ny * width) + nx];
      }
      sum += pixel_value * kGaussKernel.at(ky + 1).at(kx + 1);
    }
  }
  return static_cast<uint8_t>(std::clamp(sum / kKernelSum, 0, 255));
}
}  // namespace

MarinLLinearFilterVerticalSEQ::MarinLLinearFilterVerticalSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool MarinLLinearFilterVerticalSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.width <= 0 || input.height <= 0) {
    return false;
  }
  auto expected_size = static_cast<size_t>(input.width) * static_cast<size_t>(input.height);
  return input.pixels.size() == expected_size;
}

bool MarinLLinearFilterVerticalSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  width_ = input.width;
  height_ = input.height;
  input_pixels_ = input.pixels;
  output_pixels_.resize(input_pixels_.size());
  return true;
}

bool MarinLLinearFilterVerticalSEQ::RunImpl() {
  for (int row = 0; row < height_; ++row) {
    for (int col = 0; col < width_; ++col) {
      output_pixels_[(row * width_) + col] = ApplyKernel(input_pixels_, width_, height_, row, col);
    }
  }
  return true;
}

bool MarinLLinearFilterVerticalSEQ::PostProcessingImpl() {
  GetOutput() = output_pixels_;
  return true;
}

}  // namespace marin_l_linear_filter_vertical
