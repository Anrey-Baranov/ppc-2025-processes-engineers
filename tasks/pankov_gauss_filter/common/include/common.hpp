#pragma once

#include <cstdint>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace pankov_gauss_filter {

struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<std::uint8_t> data;

  Image() = default;

  Image(int w, int h, int ch, std::vector<std::uint8_t> d) : width(w), height(h), channels(ch), data(std::move(d)) {}
};

inline void PrintTo(const Image &img, std::ostream *os) {
  *os << "Image{width=" << img.width << ", height=" << img.height << ", channels=" << img.channels
      << ", data_size=" << img.data.size() << "}";
}

using InType = Image;
using OutType = Image;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace pankov_gauss_filter
