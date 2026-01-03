#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace vlasova_a_image_smoothing {

struct ImageData {
  std::vector<std::uint8_t> data;
  int width = 0;
  int height = 0;

  bool operator==(const ImageData &other) const {
    return width == other.width && height == other.height && data == other.data;
  }
};

using InType = ImageData;
using OutType = ImageData;
using TestType = std::tuple<int, std::string>;

}  // namespace vlasova_a_image_smoothing
