#include "pankov_gauss_filter/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "pankov_gauss_filter/common/include/common.hpp"

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

}  // namespace

PankovGaussFilterSEQ::PankovGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp(in);
  std::swap(GetInput(), temp);
  GetOutput() = OutType{};
}

bool PankovGaussFilterSEQ::ValidationImpl() {
  const auto &in = GetInput();
  if (!GetOutput().data.empty()) {
    return false;
  }
  if (in.width < 0 || in.height < 0 || in.channels < 0) {
    return false;
  }
  if (in.width == 0 || in.height == 0) {
    return in.data.empty();
  }
  if (in.channels == 0) {
    return false;
  }
  const auto expected =
      static_cast<std::size_t>(in.width) * static_cast<std::size_t>(in.height) * static_cast<std::size_t>(in.channels);
  return in.data.size() == expected;
}

bool PankovGaussFilterSEQ::PreProcessingImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  out.width = in.width;
  out.height = in.height;
  out.channels = in.channels;
  const auto total =
      static_cast<std::size_t>(in.width) * static_cast<std::size_t>(in.height) * static_cast<std::size_t>(in.channels);
  out.data.assign(total, 0);
  return true;
}

bool PankovGaussFilterSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  if (in.width == 0 || in.height == 0) {
    return true;
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

  return true;
}

bool PankovGaussFilterSEQ::PostProcessingImpl() {
  const auto &out = GetOutput();
  if (out.width == 0 || out.height == 0) {
    return out.data.empty();
  }
  return !out.data.empty();
}

}  // namespace pankov_gauss_filter
