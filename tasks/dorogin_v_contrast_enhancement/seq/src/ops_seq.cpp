#include "dorogin_v_contrast_enhancement/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "dorogin_v_contrast_enhancement/common/include/common.hpp"

namespace dorogin_v_contrast_enhancement {

DoroginVContrastEnhancementSEQ::DoroginVContrastEnhancementSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool DoroginVContrastEnhancementSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool DoroginVContrastEnhancementSEQ::PreProcessingImpl() {
  image_ = GetInput();
  result_.resize(image_.size());
  for (std::size_t i = 0; i < image_.size(); ++i) {
    result_[i] = 0;
  }
  return true;
}

bool DoroginVContrastEnhancementSEQ::RunImpl() {
  if (image_.empty()) {
    return true;
  }

  uint8_t min_val = *std::ranges::min_element(image_);
  uint8_t max_val = *std::ranges::max_element(image_);

  if (min_val == max_val) {
    result_ = image_;
    return true;
  }

  int range = max_val - min_val;
  for (std::size_t i = 0; i < image_.size(); ++i) {
    result_[i] = static_cast<uint8_t>(((image_[i] - min_val) * 255) / range);
  }

  return true;
}

bool DoroginVContrastEnhancementSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace dorogin_v_contrast_enhancement
