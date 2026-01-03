#pragma once

#include <cstdint>
#include <tuple>

#include "task/include/task.hpp"

namespace likhanov_m_elem_vec_sum {

using InType = int64_t;
using OutType = int64_t;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace likhanov_m_elem_vec_sum
