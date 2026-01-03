#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shemetov_d_increasing_contrast {

struct Pixel {
  uint8_t channel_red;
  uint8_t channel_green;
  uint8_t channel_blue;
};

using InType = std::vector<Pixel>;
using OutType = std::vector<Pixel>;
using TestType = std::tuple<std::string, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shemetov_d_increasing_contrast
