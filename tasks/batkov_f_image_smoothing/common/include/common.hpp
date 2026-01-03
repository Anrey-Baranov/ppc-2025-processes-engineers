#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace batkov_f_image_smoothing {

struct Image {
  std::vector<uint8_t> data;
  size_t width{};
  size_t height{};
  size_t channels{};
};

using InType = Image;
using OutType = Image;
using TestType = std::tuple<std::string, size_t, size_t, size_t>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace batkov_f_image_smoothing
