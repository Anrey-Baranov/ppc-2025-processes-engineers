#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace posternak_a_increase_contrast {

using InType = std::vector<unsigned char>;
using OutType = std::vector<unsigned char>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace posternak_a_increase_contrast
