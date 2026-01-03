#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"

namespace shemetov_d_increasing_contrast {

IncreaseContrastTaskSEQ::IncreaseContrastTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool IncreaseContrastTaskSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool IncreaseContrastTaskSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool IncreaseContrastTaskSEQ::RunImpl() {
  constexpr float kFactor = 1.3F;

  for (size_t i = 0; i < GetInput().size(); ++i) {
    const Pixel &local_input = GetInput()[i];
    Pixel &local_output = GetOutput()[i];

    const auto m_red = static_cast<float>(local_input.channel_red) * kFactor;
    const auto m_green = static_cast<float>(local_input.channel_red) * kFactor;
    const auto m_blue = static_cast<float>(local_input.channel_red) * kFactor;

    local_output.channel_red = static_cast<uint8_t>(std::clamp(m_red, 0.F, 255.F));
    local_output.channel_green = static_cast<uint8_t>(std::clamp(m_green, 0.F, 255.F));
    local_output.channel_blue = static_cast<uint8_t>(std::clamp(m_blue, 0.F, 255.F));
  }

  return true;
}

bool IncreaseContrastTaskSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace shemetov_d_increasing_contrast
