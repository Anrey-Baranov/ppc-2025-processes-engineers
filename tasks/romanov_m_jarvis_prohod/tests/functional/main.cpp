#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "romanov_m_jarvis_prohod/common/include/common.hpp"
#include "romanov_m_jarvis_prohod/mpi/include/ops_mpi.hpp"
#include "romanov_m_jarvis_prohod/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace romanov_m_jarvis_prohod {

class RomanovJarvisConvexHullTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return "Test_" + std::to_string(std::get<0>(param));
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_points_ = std::get<1>(params);
    expected_hull_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output) final {
    if (output.size() != expected_hull_.size()) {
      return false;
    }
    for (const auto &p : output) {
      bool found = false;
      for (const auto &src : input_points_) {
        if (p.x == src.x && p.y == src.y) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }

    for (std::size_t i = 0; i < output.size(); ++i) {
      const auto &a = output[i];
      const auto &b = output[(i + 1) % output.size()];
      const auto &c = output[(i + 2) % output.size()];
      if (CrossCalculate(a, b, c) < 0) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_points_;
  }

 private:
  std::vector<Point> input_points_;
  std::vector<Point> expected_hull_;
};

namespace {

const std::array<TestType, 4> kJarvisTests = {
    std::make_tuple(1, std::vector<Point>{{0, 0}, {2, 0}, {1, 1}}, std::vector<Point>{{0, 0}, {2, 0}, {1, 1}}),

    std::make_tuple(2, std::vector<Point>{{0, 0}, {1, 0}, {2, 0}, {3, 0}}, std::vector<Point>{{0, 0}, {3, 0}}),

    std::make_tuple(3, std::vector<Point>{{0, 0}, {0, 3}, {3, 3}, {3, 0}, {1, 1}},
                    std::vector<Point>{{0, 0}, {3, 0}, {3, 3}, {0, 3}}),

    std::make_tuple(4, std::vector<Point>{{-2, -1}, {-1, -2}, {1, -1}, {2, 2}, {0, 0}},
                    std::vector<Point>{{-1, -2}, {1, -1}, {2, 2}, {-2, -1}})};

const auto kTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<RomanovMJarvisProhodMPI, InType>(kJarvisTests, PPC_SETTINGS_romanov_m_jarvis_prohod),
    ppc::util::AddFuncTask<RomanovMJarvisProhodSEQ, InType>(kJarvisTests, PPC_SETTINGS_romanov_m_jarvis_prohod));

const auto kGtestValues = ppc::util::ExpandToValues(kTasksList);

const auto kTestName = RomanovJarvisConvexHullTests::PrintFuncTestName<RomanovJarvisConvexHullTests>;

TEST_P(RomanovJarvisConvexHullTests, ConvexHullCorrectness) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(JarvisAlgorithmTests, RomanovJarvisConvexHullTests, kGtestValues, kTestName);

TEST(RomanovJarvisValidation, MpiFailsForSmallInput) {
  InType points = {{0, 0}, {1, 1}};
  RomanovMJarvisProhodMPI task(points);
  EXPECT_FALSE(task.Validation());
}

TEST(RomanovJarvisValidation, SeqFailsForSinglePoint) {
  InType points = {{0, 0}};
  RomanovMJarvisProhodSEQ task(points);
  EXPECT_FALSE(task.Validation());
}

}  // namespace
}  // namespace romanov_m_jarvis_prohod
