#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace belov_e_shell_batcher {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace belov_e_shell_batcher
