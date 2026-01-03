#pragma once

#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace luchnikov_e_graham_cov_hall_constr {

struct Point {
  double x = 0.0;
  double y = 0.0;

  explicit Point(double x_coord = 0, double y_coord = 0) : x(x_coord), y(y_coord) {}

  bool operator<(const Point &other) const {
    return std::tie(y, x) < std::tie(other.y, other.x);
  }

  bool operator==(const Point &other) const {
    static constexpr double kEpsilon = 1e-9;
    return std::abs(x - other.x) < kEpsilon && std::abs(y - other.y) < kEpsilon;
  }
};

using InType = std::vector<Point>;
using OutType = std::vector<Point>;
using TestType = std::tuple<InType, OutType>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace luchnikov_e_graham_cov_hall_constr
