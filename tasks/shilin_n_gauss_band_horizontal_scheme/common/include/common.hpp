#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shilin_n_gauss_band_horizontal_scheme {

using InType = std::vector<std::vector<double>>;  // Ленточная матрица A и вектор b
using OutType = std::vector<double>;              // Вектор решения x
using TestType = std::tuple<int, int>;            // Размер матрицы и ширина ленты
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shilin_n_gauss_band_horizontal_scheme
