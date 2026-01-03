#include "zyazeva_s_graham_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "zyazeva_s_graham_scheme/common/include/common.hpp"

namespace zyazeva_s_graham_scheme {

namespace {

int Cross(const Point &origin, const Point &a, const Point &b) {
  return ((a.x - origin.x) * (b.y - origin.y)) - ((a.y - origin.y) * (b.x - origin.x));
}

std::vector<Point> BuildConvexHull(std::vector<Point> pts) {
  std::ranges::sort(pts.begin(), pts.end(),
                    [](const Point &a, const Point &b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });

  std::vector<Point> hull;

  for (const auto &p : pts) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  std::size_t lower_size = hull.size();
  for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
    const auto &p = pts[i];
    while (hull.size() > lower_size && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  hull.pop_back();
  return hull;
}

}  // namespace

ZyazevaSGrahamSchemeSEQ::ZyazevaSGrahamSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ZyazevaSGrahamSchemeSEQ::ValidationImpl() {
  if (GetInput().size() < 3) {
    GetOutput().clear();
  }
  return true;
}

bool ZyazevaSGrahamSchemeSEQ::PreProcessingImpl() {
  return true;
}

bool ZyazevaSGrahamSchemeSEQ::RunImpl() {
  const auto &points = GetInput();

  if (points.size() < 3) {
    GetOutput().clear();
    return true;
  }

  GetOutput() = BuildConvexHull(points);
  return true;
}

bool ZyazevaSGrahamSchemeSEQ::PostProcessingImpl() {
  return true;  // 1
}

}  // namespace zyazeva_s_graham_scheme
