#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace klimenko_v_multistep_2d_parallel_sad {

struct OptimizationInput {
  std::function<double(double, double)> func;

  double x_min{0.0};
  double x_max{1.0};
  double y_min{0.0};
  double y_max{1.0};

  double epsilon{0.01};
  double r_param{2.0};
  int max_iterations{1000};

  OptimizationInput() : func(nullptr) {}
};

struct OptimizationResult {
  double x_opt{0.0};
  double y_opt{0.0};
  double func_min{0.0};

  int iterations{0};
  bool converged{false};

  bool operator==(const OptimizationResult &other) const {
    const double tol = 1e-3;
    return std::abs(x_opt - other.x_opt) < tol && std::abs(y_opt - other.y_opt) < tol &&
           std::abs(func_min - other.func_min) < tol;
  }
};

struct Region {
  double x_min, x_max;
  double y_min, y_max;

  double f_center;
  double characteristic;
};

using InType = OptimizationInput;
using OutType = OptimizationResult;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace klimenko_v_multistep_2d_parallel_sad
