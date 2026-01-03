#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace timofeev_n_ribbon_scheme_only_a {

using MatrixType = std::vector<std::vector<int>>;  // очевидно для чего
using InType = std::pair<MatrixType, MatrixType>;
using OutType = MatrixType;
using TestType = std::tuple<MatrixType, OutType, InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace timofeev_n_ribbon_scheme_only_a
