#include "romanov_m_jarvis_prohod/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "romanov_m_jarvis_prohod/common/include/common.hpp"

namespace romanov_m_jarvis_prohod {

RomanovMJarvisProhodSEQ::RomanovMJarvisProhodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RomanovMJarvisProhodSEQ::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool RomanovMJarvisProhodSEQ::PreProcessingImpl() {
  return true;
}

namespace {

int LeftPoint(const std::vector<Point> &points) {
  int idx = 0;
  for (std::size_t i = 1; i < points.size(); ++i) {
    if (points[i].x < points[idx].x || (points[i].x == points[idx].x && points[i].y < points[idx].y)) {
      idx = static_cast<int>(i);
    }
  }
  return idx;
}

int ChooseNextBoundaryPoint(const std::vector<Point> &points, int p) {
  const int n = static_cast<int>(points.size());
  int q = (p + 1) % n;

  for (int i = 0; i < n; ++i) {
    const int64_t cross = CrossCalculate(points[p], points[i], points[q]);
    if (cross > 0) {
      q = i;
    } else if (cross == 0) {
      if (SqDistance(points[p], points[i]) > SqDistance(points[p], points[q])) {
        q = i;
      }
    }
  }
  return q;
}

}  // namespace

std::vector<Point> RomanovMJarvisProhodSEQ::JarvisMarch(std::vector<Point> points) {
  if (points.size() < 3) {
    return points;
  }

  std::vector<Point> hull;
  const int start = LeftPoint(points);

  int p = start;
  while (true) {
    hull.push_back(points[p]);
    p = ChooseNextBoundaryPoint(points, p);
    if (p == start) {
      break;
    }
  }

  return hull;
}

bool RomanovMJarvisProhodSEQ::RunImpl() {
  GetOutput() = JarvisMarch(GetInput());
  return true;
}

bool RomanovMJarvisProhodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_m_jarvis_prohod
