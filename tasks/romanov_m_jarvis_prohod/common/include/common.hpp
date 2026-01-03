#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace romanov_m_jarvis_prohod {

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

constexpr bool IsEqual(const Point &p1, const Point &p2) noexcept {
  return p1.x == p2.x && p1.y == p2.y;
}

constexpr int64_t CrossCalculate(const Point &p1, const Point &p2, const Point &p3) noexcept {
  return (static_cast<int64_t>(p2.x - p1.x) * static_cast<int64_t>(p3.y - p1.y)) -
         (static_cast<int64_t>(p2.y - p1.y) * static_cast<int64_t>(p3.x - p1.x));
}

constexpr int SqDistance(const Point &p1, const Point &p2) noexcept {
  const int dx = p2.x - p1.x;
  const int dy = p2.y - p1.y;
  return (dx * dx) + (dy * dy);
}

inline Point FindStartPoint(const std::vector<Point> &points) {
  return *std::ranges::min_element(
      points, [](const Point &a, const Point &b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
}

using InType = std::vector<Point>;
using OutType = std::vector<Point>;
using TestType = std::tuple<int, std::vector<Point>, std::vector<Point>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanov_m_jarvis_prohod
