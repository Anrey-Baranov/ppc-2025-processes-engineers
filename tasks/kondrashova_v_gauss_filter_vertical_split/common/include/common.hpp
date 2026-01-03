#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

struct ImageData {
  std::vector<uint8_t> pixels;
  int width{0};
  int height{0};
  int channels{1};

  bool operator==(const ImageData &other) const {
    return width == other.width && height == other.height && channels == other.channels && pixels == other.pixels;
  }
};

using InType = ImageData;
using OutType = ImageData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kondrashova_v_gauss_filter_vertical_split
