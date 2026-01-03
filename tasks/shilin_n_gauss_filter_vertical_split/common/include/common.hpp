#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shilin_n_gauss_filter_vertical_split {

struct ImageData {
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
  int channels = 0;
};

using InType = ImageData;
using OutType = std::vector<uint8_t>;
using TestType = std::tuple<int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shilin_n_gauss_filter_vertical_split
