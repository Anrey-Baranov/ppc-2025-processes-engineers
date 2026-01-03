#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace alekseev_a_global_opt_chars {

struct OptInput {
  std::function<double(double, double)> func;

  double x_min{0.0};
  double x_max{1.0};
  double y_min{0.0};
  double y_max{1.0};

  double epsilon{0.01};
  double r_param{2.0};
  int max_iterations{1000};

  OptInput() : func(nullptr) {}
};

struct OptResult {
  double x_opt{0.0};
  double y_opt{0.0};
  double func_min{0.0};
  int iterations{0};
  bool converged{false};

  OptResult() = default;

  bool operator==(const OptResult &other) const {
    constexpr double kTol = 1e-3;
    return std::abs(x_opt - other.x_opt) < kTol && std::abs(y_opt - other.y_opt) < kTol &&
           std::abs(func_min - other.func_min) < kTol;
  }
};

struct TrialPoint {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  TrialPoint() = default;
  TrialPoint(double px, double py, double pz) : x(px), y(py), z(pz) {}

  bool operator<(const TrialPoint &other) const {
    if (x != other.x) {
      return x < other.x;
    }
    return y < other.y;
  }
};

struct Interval {
  int left_idx{0};
  int right_idx{0};
  double characteristic{0.0};

  Interval() = default;
  Interval(int l, int r, double c) : left_idx(l), right_idx(r), characteristic(c) {}
};

using InType = OptInput;
using OutType = OptResult;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline double PeanoToX(double t, double x_min, double x_max, double y_min, double y_max, int level);

inline double PeanoToY(double t, double x_min, double x_max, double y_min, double y_max, int level);

namespace detail {

inline void PeanoMapUnitSquare(double t, int level, double &x, double &y) {
  x = 0.0;
  y = 0.0;

  double scale = 0.5;

  for (int i = 0; i < level; ++i) {
    int quadrant = static_cast<int>(t * 4.0);
    t = (t * 4.0) - quadrant;

    double dx = 0.0;
    double dy = 0.0;

    switch (quadrant) {
      case 0:
        dx = 0.0;
        dy = 0.0;
        break;
      case 1:
        dx = 0.0;
        dy = 1.0;
        break;
      case 2:
        dx = 1.0;
        dy = 1.0;
        break;
      case 3:
        dx = 1.0;
        dy = 0.0;
        break;
      default:
        dx = 0.0;
        dy = 0.0;
        break;
    }

    x += dx * scale;
    y += dy * scale;
    scale *= 0.5;
  }
}

}  // namespace detail

inline double PeanoToX(double t, double x_min, double x_max, int level) {
  double ux = 0.0;
  double uy = 0.0;
  detail::PeanoMapUnitSquare(t, level, ux, uy);
  return x_min + (ux * (x_max - x_min));
}

inline double PeanoToY(double t, double y_min, double y_max, int level) {
  double ux = 0.0;
  double uy = 0.0;
  detail::PeanoMapUnitSquare(t, level, ux, uy);
  return y_min + (uy * (y_max - y_min));
}

}  // namespace alekseev_a_global_opt_chars
