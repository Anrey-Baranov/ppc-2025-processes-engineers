#include "shemetov_d_gauss_filter_linear/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"

namespace shemetov_d_gauss_filter_linear {

GaussFilterSEQ::GaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

Pixel GaussFilterSEQ::ApplyKernel(const InType &in, int i, int j, const std::vector<std::vector<float>> &kernel) {
  float channel_red = 0.F;
  float channel_green = 0.F;
  float channel_blue = 0.F;

  for (int ki = -1; ki <= 1; ++ki) {
    for (int kj = -1; kj <= 1; ++kj) {
      const auto &lnk_pixel = in[i + ki][j + kj];
      float coefficient = kernel[ki + 1][kj + 1];

      channel_red += coefficient * static_cast<float>(lnk_pixel.channel_red);
      channel_green += coefficient * static_cast<float>(lnk_pixel.channel_green);
      channel_blue += coefficient * static_cast<float>(lnk_pixel.channel_blue);
    }
  }

  Pixel m_pixel = {.channel_red = static_cast<uint8_t>(std::clamp(channel_red, 0.F, 255.F)),
                   .channel_green = static_cast<uint8_t>(std::clamp(channel_green, 0.F, 255.F)),
                   .channel_blue = static_cast<uint8_t>(std::clamp(channel_blue, 0.F, 255.F))};
  return m_pixel;
}

bool GaussFilterSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return !in.empty() && !in[0].empty();
}

bool GaussFilterSEQ::PreProcessingImpl() {
  return true;
}

bool GaussFilterSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  height = static_cast<int>(in.size());
  width = static_cast<int>(in[0].size());

  if (height < 3 || width < 3) {
    out = in;
    return true;
  }

  const std::vector<std::vector<float>> kernel = {
      {1.F / 16, 2.F / 16, 1.F / 16}, {2.F / 16, 4.F / 16, 2.F / 16}, {1.F / 16, 2.F / 16, 1.F / 16}};

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      out[i][j] = ApplyKernel(in, i, j, kernel);
    }
  }

  return true;
}

bool GaussFilterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_gauss_filter_linear
