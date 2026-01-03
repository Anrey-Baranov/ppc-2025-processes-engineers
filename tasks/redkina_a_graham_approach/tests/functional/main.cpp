#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"
#include "redkina_a_graham_approach/mpi/include/ops_mpi.hpp"
#include "redkina_a_graham_approach/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace redkina_a_graham_approach {

using TestType = std::tuple<int, std::vector<Point>, std::vector<Point>>;

class RedkinaAGrahamApproachFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestName(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    const auto &task_name = std::get<1>(info.param);
    const auto &test_param = std::get<2>(info.param);
    const int test_id = std::get<0>(test_param);
    return task_name + "_Test" + std::to_string(test_id) + "_Idx" + std::to_string(info.index);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_points_ = std::get<1>(params);
    expected_hull_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_hull_.size()) {
      return false;
    }

    for (const auto &expected_point : expected_hull_) {
      bool found = false;
      for (const auto &output_point : output_data) {
        if (output_point.x == expected_point.x && output_point.y == expected_point.y) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return test_points_;
  }

 private:
  std::vector<Point> test_points_;
  std::vector<Point> expected_hull_;
};

namespace {

const std::array<TestType, 3> kFunctionalTests = {
    std::make_tuple(1, std::vector<Point>{{0, 7}, {7, 0}, {7, 7}}, std::vector<Point>{{0, 7}, {7, 0}, {7, 7}}),

    std::make_tuple(2, std::vector<Point>{{0, 0}, {7, 0}, {7, 7}, {0, 7}},
                    std::vector<Point>{{0, 0}, {7, 0}, {7, 7}, {0, 7}}),

    std::make_tuple(3, std::vector<Point>{{7, 7}, {14, 14}, {21, 21}, {28, 28}}, std::vector<Point>{{7, 7}, {28, 28}})};

const std::array<TestType, 3> kCoverageTests = {
    std::make_tuple(4, std::vector<Point>{{7, 7}, {14, 7}, {7, 14}}, std::vector<Point>{{7, 7}, {14, 7}, {7, 14}}),

    std::make_tuple(5, std::vector<Point>{{-7, -7}, {7, -7}, {0, 7}}, std::vector<Point>{{-7, -7}, {7, -7}, {0, 7}}),

    std::make_tuple(6, std::vector<Point>{{7, 7}, {21, 7}, {14, 21}}, std::vector<Point>{{7, 7}, {21, 7}, {14, 21}})};

const auto kFunctionalTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<RedkinaAGrahamApproachMPI, InType>(kFunctionalTests, PPC_SETTINGS_redkina_a_graham_approach),
    ppc::util::AddFuncTask<RedkinaAGrahamApproachSEQ, InType>(kFunctionalTests,
                                                              PPC_SETTINGS_redkina_a_graham_approach));

const auto kCoverageTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<RedkinaAGrahamApproachMPI, InType>(kCoverageTests, PPC_SETTINGS_redkina_a_graham_approach),
    ppc::util::AddFuncTask<RedkinaAGrahamApproachSEQ, InType>(kCoverageTests, PPC_SETTINGS_redkina_a_graham_approach));

inline const auto kFunctionalGtestValues = ppc::util::ExpandToValues(kFunctionalTasksList);
inline const auto kCoverageGtestValues = ppc::util::ExpandToValues(kCoverageTasksList);

TEST_P(RedkinaAGrahamApproachFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

TEST_P(RedkinaAGrahamApproachFuncTests, CoverageTests) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Functional, RedkinaAGrahamApproachFuncTests, kFunctionalGtestValues,
                         RedkinaAGrahamApproachFuncTests::PrintTestName);

INSTANTIATE_TEST_SUITE_P(Coverage, RedkinaAGrahamApproachFuncTests, kCoverageGtestValues,
                         RedkinaAGrahamApproachFuncTests::PrintTestName);

TEST(RedkinaAGrahamApproachValidation, MpiValidationFailsForLessThan3Points) {
  InType points = {{77, 77}, {7, 7}};
  RedkinaAGrahamApproachMPI task(points);
  EXPECT_FALSE(task.Validation());
}

TEST(RedkinaAGrahamApproachValidation, SeqValidationFailsForLessThan3Points) {
  InType points = {{7, 7}};
  RedkinaAGrahamApproachSEQ task(points);
  EXPECT_FALSE(task.Validation());
}

}  // namespace
}  // namespace redkina_a_graham_approach
