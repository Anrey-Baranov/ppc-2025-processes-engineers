#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace rastvorov_k_number_of_character_alternations {

using InType = double;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace rastvorov_k_number_of_character_alternations
