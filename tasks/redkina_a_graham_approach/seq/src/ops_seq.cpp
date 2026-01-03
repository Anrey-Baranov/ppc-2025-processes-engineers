#include "redkina_a_graham_approach/seq/include/ops_seq.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"

namespace redkina_a_graham_approach {

RedkinaAGrahamApproachSEQ::RedkinaAGrahamApproachSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RedkinaAGrahamApproachSEQ::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool RedkinaAGrahamApproachSEQ::PreProcessingImpl() {
  return true;
}

namespace {

constexpr bool ComparePolarAngles(const Point &pivot, const Point &a, const Point &b) noexcept {
  const int cross = ((a.x - pivot.x) * (b.y - pivot.y)) - ((a.y - pivot.y) * (b.x - pivot.x));
  if (cross == 0) {
    const int dx1 = a.x - pivot.x;
    const int dy1 = a.y - pivot.y;
    const int dx2 = b.x - pivot.x;
    const int dy2 = b.y - pivot.y;
    return ((dx1 * dx1) + (dy1 * dy1)) < ((dx2 * dx2) + (dy2 * dy2));
  }
  return cross > 0;
}

std::vector<Point> GrahamScanSeq(std::vector<Point> points) {
  if (points.size() < 3) {
    return points;
  }

  const auto pivot_it = std::ranges::min_element(
      points, [](const Point &a, const Point &b) { return a.y < b.y || (a.y == b.y && a.x < b.x); });
  std::swap(points.front(), *pivot_it);
  const Point pivot = points.front();

  std::ranges::sort(points.begin() + 1, points.end(),
                    [&pivot](const Point &a, const Point &b) { return ComparePolarAngles(pivot, a, b); });

  std::vector<Point> hull;
  hull.reserve(points.size());
  for (const auto &p : points) {
    while (hull.size() >= 2 && CalcCross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }
  return hull;
}

}  // namespace

bool RedkinaAGrahamApproachSEQ::RunImpl() {
  auto pts = GetInput();
  auto res = GrahamScanSeq(std::move(pts));
  if (res.empty() && !GetInput().empty()) {
    res.push_back(GetInput().front());
  }
  GetOutput() = std::move(res);
  return true;
}

bool RedkinaAGrahamApproachSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_graham_approach
