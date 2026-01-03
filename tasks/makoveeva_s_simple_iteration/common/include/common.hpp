#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace makoveeva_s_simple_iteration {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

using TypeOfTask = ppc::task::TypeOfTask;

}  // namespace makoveeva_s_simple_iteration
