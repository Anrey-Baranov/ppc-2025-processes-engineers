#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace redkina_a_graham_approach {

struct Point {
  int x{};
  int y{};

  constexpr bool operator==(const Point &other) const noexcept {
    return x == other.x && y == other.y;
  }

  constexpr bool operator!=(const Point &other) const noexcept {
    return !(*this == other);
  }
};

constexpr bool ArePointsEqual(const Point &p1, const Point &p2) noexcept {
  return p1.x == p2.x && p1.y == p2.y;
}

constexpr int CalcCross(const Point &p1, const Point &p2, const Point &p3) noexcept {
  return ((p2.x - p1.x) * (p3.y - p1.y)) - ((p2.y - p1.y) * (p3.x - p1.x));
}

constexpr int CalcDistSq(const Point &p1, const Point &p2) noexcept {
  const int dx = p2.x - p1.x;
  const int dy = p2.y - p1.y;
  return (dx * dx) + (dy * dy);
}

inline Point FindPivotPoint(const std::vector<Point> &points) {
  return *std::ranges::min_element(
      points, [](const Point &a, const Point &b) { return a.y < b.y || (a.y == b.y && a.x < b.x); });
}

using InType = std::vector<Point>;
using OutType = std::vector<Point>;
using TestType = std::tuple<int, std::vector<Point>, std::vector<Point>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace redkina_a_graham_approach
