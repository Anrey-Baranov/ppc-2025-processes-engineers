#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace pankov_matrix_vector {

struct MatrixVectorInput {
  std::vector<std::vector<double>> matrix;
  std::vector<double> vector;
};

using InType = MatrixVectorInput;
using OutType = std::vector<double>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace pankov_matrix_vector
