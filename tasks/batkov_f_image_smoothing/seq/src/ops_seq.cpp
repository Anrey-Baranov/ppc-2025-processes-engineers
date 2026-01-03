#include "batkov_f_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "batkov_f_image_smoothing/common/include/common.hpp"

namespace batkov_f_image_smoothing {

BatkovFImageSmoothingSEQ::BatkovFImageSmoothingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Image();
}

bool BatkovFImageSmoothingSEQ::ValidationImpl() {
  return (!GetInput().data.empty()) && (GetInput().width > 0) && (GetInput().height > 0);
}

bool BatkovFImageSmoothingSEQ::PreProcessingImpl() {
  size_t size = 5;
  float sigma = 1.0F;

  gaussian_kernel_.resize(size);
  for (auto &v : gaussian_kernel_) {
    v.resize(size);
  }

  float sum = 0.0F;
  size_t half = size / 2;

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      size_t x = i - half;
      size_t y = j - half;
      float value = std::exp((-static_cast<float>((x * x) + (y * y)) / (2 * sigma * sigma)));
      gaussian_kernel_[i][j] = value;
      sum += value;
    }
  }

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      gaussian_kernel_[i][j] /= sum;
    }
  }

  return true;
}

bool BatkovFImageSmoothingSEQ::RunImpl() {
  auto &img = GetInput();
  size_t width = img.width;
  size_t height = img.height;
  size_t channels = img.channels;
  const auto &img_data = img.data;

  std::vector<uint8_t> temp(width * height * channels);

  size_t kernel_size = 5;
  size_t half = kernel_size / 2;

  for (size_t y_px = 0; y_px < height; y_px++) {
    for (size_t x_px = 0; x_px < width; x_px++) {
      for (size_t ch = 0; ch < channels; ch++) {
        float value = 0.0F;

        for (size_t ky = 0; ky < kernel_size; ky++) {
          for (size_t kx = 0; kx < kernel_size; kx++) {
            size_t px = x_px + kx - half;
            size_t py = y_px + ky - half;

            px = std::max<size_t>(0, std::min(px, width - 1));
            py = std::max<size_t>(0, std::min(py, height - 1));

            uint8_t pixel_value = img_data[((py * width + px) * channels) + ch];
            value += static_cast<float>(pixel_value) * gaussian_kernel_[ky][kx];
          }
        }

        temp[((y_px * width + x_px) * channels) + ch] = static_cast<uint8_t>(value);
      }
    }
  }

  GetOutput().data = std::move(temp);
  GetOutput().width = width;
  GetOutput().height = height;
  GetOutput().channels = channels;
  return true;
}

bool BatkovFImageSmoothingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace batkov_f_image_smoothing
