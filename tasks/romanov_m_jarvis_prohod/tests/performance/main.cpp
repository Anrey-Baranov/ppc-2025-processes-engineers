#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "romanov_m_jarvis_prohod/common/include/common.hpp"
#include "romanov_m_jarvis_prohod/mpi/include/ops_mpi.hpp"
#include "romanov_m_jarvis_prohod/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace romanov_m_jarvis_prohod {

static bool CheckValidHull(const std::vector<Point> &points, const std::vector<Point> &hull) {
  if (hull.empty()) {
    return points.empty();
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
    if (CrossCalculate(p1, p2, p3) < 0) {
      return false;
    }
  }

  return true;
}

class RomanovMJarvisProhodRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr std::size_t kSize = 1000000;

 protected:
  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000, 1000);

    i_points_.resize(kSize);
    for (std::size_t i = 0; i < kSize; ++i) {
      i_points_[i] = Point{.x = dist(gen), .y = dist(gen)};
    }

    i_points_[0] = Point{.x = -30000, .y = 0};
    i_points_[1] = Point{.x = 30000, .y = 0};
    i_points_[2] = Point{.x = 0, .y = -30000};
    i_points_[3] = Point{.x = 0, .y = 30000};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return CheckValidHull(i_points_, output_data);
  }

  InType GetTestInputData() final {
    return i_points_;
  }

 private:
  InType i_points_;
};

TEST_P(RomanovMJarvisProhodRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, RomanovMJarvisProhodMPI, RomanovMJarvisProhodSEQ>(
    PPC_SETTINGS_romanov_m_jarvis_prohod);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovMJarvisProhodRunPerfTests, kGtestValues,
                         RomanovMJarvisProhodRunPerfTests::CustomPerfTestName);

}  // namespace romanov_m_jarvis_prohod
