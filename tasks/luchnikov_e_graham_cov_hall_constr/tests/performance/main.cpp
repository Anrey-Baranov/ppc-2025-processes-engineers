#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "luchnikov_e_graham_cov_hall_constr/common/include/common.hpp"
#include "luchnikov_e_graham_cov_hall_constr/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_graham_cov_hall_constr/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_graham_cov_hall_constr {

namespace {

constexpr std::size_t kPointCount = 1'000'000;
constexpr int kCoordinateMin = 0;
constexpr int kCoordinateMax = 10'000;
constexpr int kMaxGenerationAttempts = 100;
constexpr double kEpsilon = 1e-6;

struct PairHash {
  std::size_t operator()(const std::pair<int, int> &p) const noexcept {
    const std::size_t h1 = std::hash<int>{}(p.first);
    const std::size_t h2 = std::hash<int>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

}  // namespace

class LuchnikovEGrahamConvexHullPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<int> dist(kCoordinateMin, kCoordinateMax);

    std::vector<Point> points;
    points.reserve(kPointCount);

    std::unordered_set<std::pair<int, int>, PairHash> unique_points;
    unique_points.reserve(kPointCount);

    for (std::size_t i = 0; i < kPointCount; ++i) {
      bool inserted = false;

      for (int attempt = 0; attempt < kMaxGenerationAttempts && !inserted; ++attempt) {
        const int x = dist(gen);
        const int y = dist(gen);

        if (unique_points.emplace(x, y).second) {
          points.emplace_back(static_cast<double>(x), static_cast<double>(y));
          inserted = true;
        }
      }

      if (!inserted) {
        const int x = dist(gen);
        const int y = dist(gen);
        points.emplace_back(static_cast<double>(x), static_cast<double>(y));
      }
    }

    input_data_ = std::move(points);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty() || output_data.size() > kPointCount) {
      return false;
    }
    return std::ranges::all_of(output_data, [](const Point &point) {
      return point.x >= static_cast<double>(kCoordinateMin) - kEpsilon &&
             point.x <= static_cast<double>(kCoordinateMax) + kEpsilon &&
             point.y >= static_cast<double>(kCoordinateMin) - kEpsilon &&
             point.y <= static_cast<double>(kCoordinateMax) + kEpsilon;
    });
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(LuchnikovEGrahamConvexHullPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnikovEGrahamConvexHullMPI, LuchnikovEGrahamConvexHullSEQ>(
        PPC_SETTINGS_luchnikov_e_graham_cov_hall_constr);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnikovEGrahamConvexHullPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnikovEGrahamConvexHullPerfTests, kGtestValues, kPerfTestName);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LuchnikovEGrahamConvexHullPerfTests);

}  // namespace luchnikov_e_graham_cov_hall_constr
