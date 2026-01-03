#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace ovchinnikov_m_multidim_integrals_rectangle {

// Входной тип: tuple(количество разбиений, размерность, вектор нижних границ, вектор верхних границ)
using InType = std::tuple<int, int, std::vector<double>, std::vector<double>>;
// Выходной тип: результат интегрирования
using OutType = double;
using TestType = std::tuple<int, int, std::vector<double>, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ovchinnikov_m_multidim_integrals_rectangle
