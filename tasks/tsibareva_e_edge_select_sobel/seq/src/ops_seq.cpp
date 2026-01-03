#include "tsibareva_e_edge_select_sobel/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "tsibareva_e_edge_select_sobel/common/include/common.hpp"

namespace tsibareva_e_edge_select_sobel {

const std::vector<std::vector<int>> kSobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

const std::vector<std::vector<int>> kSobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

TsibarevaEEdgeSelectSobelSEQ::TsibarevaEEdgeSelectSobelSEQ(const InType &in)
    : height_(std::get<1>(in)), width_(std::get<2>(in)), threshold_(std::get<3>(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TsibarevaEEdgeSelectSobelSEQ::ValidationImpl() {
  return true;
}

bool TsibarevaEEdgeSelectSobelSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<int>(static_cast<size_t>(height_ * width_), 0);
  return true;
}

bool TsibarevaEEdgeSelectSobelSEQ::RunImpl() {
  const auto &flat_pixels = std::get<0>(GetInput());
  input_pixels_ = std::vector<int>(flat_pixels);

  auto &output_pixels = GetOutput();

  for (int row = 0; row < height_; ++row) {
    for (int col = 0; col < width_; ++col) {
      int gx = GradientX(col, row);
      int gy = GradientY(col, row);

      int mag = static_cast<int>(std::sqrt((gx * gx) + (gy * gy) + 0.0));
      output_pixels[(static_cast<size_t>(row) * width_) + col] = (mag <= threshold_) ? 0 : mag;
    }
  }
  return true;
}

bool TsibarevaEEdgeSelectSobelSEQ::PostProcessingImpl() {
  return true;
}

int TsibarevaEEdgeSelectSobelSEQ::GradientX(int x, int y) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;

      int weight = kSobelX[ky + 1][kx + 1];

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
        sum += weight * input_pixels_[(static_cast<size_t>(ny) * width_) + nx];
      }
    }
  }

  return sum;
}

int TsibarevaEEdgeSelectSobelSEQ::GradientY(int x, int y) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;

      int weight = kSobelY[ky + 1][kx + 1];

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
        sum += weight * input_pixels_[(static_cast<size_t>(ny) * width_) + nx];
      }
    }
  }

  return sum;
}

}  // namespace tsibareva_e_edge_select_sobel
