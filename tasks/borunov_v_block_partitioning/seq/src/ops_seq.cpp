#include "borunov_v_block_partitioning/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "borunov_v_block_partitioning/common/include/common.hpp"

namespace borunov_v_block_partitioning {

BorunovVBlockPartitioningSEQ::BorunovVBlockPartitioningSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BorunovVBlockPartitioningSEQ::ValidationImpl() {
  if (GetInput().size() < 2) {
    return false;
  }
  int w = GetInput()[0];
  int h = GetInput()[1];
  const std::size_t pixels = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  const std::size_t expected_size = static_cast<std::size_t>(2) + pixels;
  return GetInput().size() == expected_size;
}

bool BorunovVBlockPartitioningSEQ::PreProcessingImpl() {
  int w = GetInput()[0];
  int h = GetInput()[1];
  GetOutput().assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  return true;
}

bool BorunovVBlockPartitioningSEQ::RunImpl() {
  int width = GetInput()[0];
  int height = GetInput()[1];
  const int *pixels = GetInput().data() + 2;

  const std::array<std::array<float, 3>, 3> kernel = {{
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
      {2.0F / 16.0F, 4.0F / 16.0F, 2.0F / 16.0F},
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
  }};

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int x0 = std::clamp(j - 1, 0, width - 1);
      const int x1 = j;
      const int x2 = std::clamp(j + 1, 0, width - 1);

      const int y0 = std::clamp(i - 1, 0, height - 1);
      const int y1 = i;
      const int y2 = std::clamp(i + 1, 0, height - 1);

      float sum = 0.0F;

      sum += static_cast<float>(pixels[(y0 * width) + x0]) * kernel[0][0];
      sum += static_cast<float>(pixels[(y0 * width) + x1]) * kernel[0][1];
      sum += static_cast<float>(pixels[(y0 * width) + x2]) * kernel[0][2];

      sum += static_cast<float>(pixels[(y1 * width) + x0]) * kernel[1][0];
      sum += static_cast<float>(pixels[(y1 * width) + x1]) * kernel[1][1];
      sum += static_cast<float>(pixels[(y1 * width) + x2]) * kernel[1][2];

      sum += static_cast<float>(pixels[(y2 * width) + x0]) * kernel[2][0];
      sum += static_cast<float>(pixels[(y2 * width) + x1]) * kernel[2][1];
      sum += static_cast<float>(pixels[(y2 * width) + x2]) * kernel[2][2];

      GetOutput()[(i * width) + j] = static_cast<int>(std::round(sum));
    }
  }
  return true;
}

bool BorunovVBlockPartitioningSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_block_partitioning
