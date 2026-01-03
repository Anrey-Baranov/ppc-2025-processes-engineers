#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace marin_l_linear_filter_vertical {

struct ImageData {
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
};

using InType = ImageData;
using OutType = std::vector<uint8_t>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace marin_l_linear_filter_vertical
