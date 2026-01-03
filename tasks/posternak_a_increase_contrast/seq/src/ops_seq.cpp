#include "posternak_a_increase_contrast/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "posternak_a_increase_contrast/common/include/common.hpp"

namespace posternak_a_increase_contrast {

PosternakAIncreaseContrastSEQ::PosternakAIncreaseContrastSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool PosternakAIncreaseContrastSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool PosternakAIncreaseContrastSEQ::PreProcessingImpl() {
  return true;
}

bool PosternakAIncreaseContrastSEQ::RunImpl() {
  const std::vector<unsigned char> &input = GetInput();
  std::vector<unsigned char> &output = GetOutput();

  unsigned char min_val = *std::ranges::min_element(input);
  unsigned char max_val = *std::ranges::max_element(input);

  if (min_val == max_val) {
    std::ranges::fill(output, 128);
    return true;
  }

  double scale = 255.0 / static_cast<double>(max_val - min_val);

  for (size_t i = 0; i < input.size(); ++i) {
    double new_pixel = static_cast<double>(input[i] - min_val) * scale;
    new_pixel = std::round(new_pixel);
    new_pixel = std::max(new_pixel, 0.0);
    new_pixel = std::min(new_pixel, 255.0);

    output[i] = static_cast<unsigned char>(new_pixel);
  }

  return true;
}

bool PosternakAIncreaseContrastSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace posternak_a_increase_contrast
