#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"
#include "redkina_a_graham_approach/mpi/include/ops_mpi.hpp"
#include "redkina_a_graham_approach/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace redkina_a_graham_approach {

static bool IsValidConvexHull(const std::vector<Point> &points, const std::vector<Point> &hull) {
  if (hull.empty()) {
    return points.empty();
  }

  if (hull.size() == 1) {
    return points.size() == 1 && points[0] == hull[0];
  }

  if (hull.size() == 2) {
    return (std::ranges::find(points, hull[0]) != points.end()) && (std::ranges::find(points, hull[1]) != points.end());
  }

  for (const auto &h : hull) {
    if (std::ranges::find(points, h) == points.end()) {
      return false;
    }
  }

  for (std::size_t i = 0; i < hull.size(); ++i) {
    const Point &p1 = hull[i];
    const Point &p2 = hull[(i + 1) % hull.size()];
    const Point &p3 = hull[(i + 2) % hull.size()];

    int cross = ((p2.x - p1.x) * (p3.y - p1.y)) - ((p2.y - p1.y) * (p3.x - p1.x));
    if (cross < 0) {
      return false;
    }
  }

  for (const auto &p : points) {
    for (std::size_t i = 0; i < hull.size(); ++i) {
      const Point &a = hull[i];
      const Point &b = hull[(i + 1) % hull.size()];

      int cross = ((b.x - a.x) * (p.y - a.y)) - ((b.y - a.y) * (p.x - a.x));
      if (cross < 0) {
        return false;
      }
      if (cross == 0) {
        int min_x = std::min(a.x, b.x);
        int max_x = std::max(a.x, b.x);
        int min_y = std::min(a.y, b.y);
        int max_y = std::max(a.y, b.y);
        if (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y) {
          return false;
        }
      }
    }
  }

  return true;
}

class RedkinaAGrahamApproachRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr std::size_t kSize = 1000000;

 protected:
  void SetUp() override {
    std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000, 1000);

    i_points_.resize(kSize);
    for (std::size_t i = 0; i < kSize; ++i) {
      i_points_[i] = Point{.x = dist(gen), .y = dist(gen)};
    }

    i_points_[0] = Point{.x = -10000, .y = -10000};
    i_points_[1] = Point{.x = 10000, .y = -10000};
    i_points_[2] = Point{.x = 10000, .y = 10000};
    i_points_[3] = Point{.x = -10000, .y = 10000};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (!IsValidConvexHull(i_points_, output_data)) {
      return false;
    }

    std::vector<Point> extreme_pts = {Point{.x = -10000, .y = -10000}, Point{.x = 10000, .y = -10000},
                                      Point{.x = 10000, .y = 10000}, Point{.x = -10000, .y = 10000}};

    for (const auto &p : extreme_pts) {
      if (std::ranges::find(output_data, p) == output_data.end()) {
        bool found_on_edge = false;
        for (const auto &h : output_data) {
          if (h.x == -10000 || h.x == 10000 || h.y == -10000 || h.y == 10000) {
            found_on_edge = true;
            break;
          }
        }
        if (!found_on_edge) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return i_points_;
  }

 private:
  InType i_points_;
};

TEST_P(RedkinaAGrahamApproachRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, RedkinaAGrahamApproachMPI, RedkinaAGrahamApproachSEQ>(
    PPC_SETTINGS_redkina_a_graham_approach);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RedkinaAGrahamApproachRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RedkinaAGrahamApproachRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace redkina_a_graham_approach
