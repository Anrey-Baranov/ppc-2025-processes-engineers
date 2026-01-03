#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "luchnikov_e_graham_cov_hall_constr/common/include/common.hpp"
#include "luchnikov_e_graham_cov_hall_constr/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_graham_cov_hall_constr/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_graham_cov_hall_constr {

namespace {

constexpr double kEpsilon = 1e-9;

constexpr TestType CreateTestData(const InType &input, const OutType &expected) {
  return std::make_tuple(input, expected);
}

}  // namespace

class LuchnikovEGrahamConvexHullFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam([[maybe_unused]] const TestType &test_param) {
    static std::size_t test_counter = 0;
    ++test_counter;
    return "Test_" + std::to_string(test_counter);
  }

 protected:
  void SetUp() override {
    const TestType &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    for (const Point &point : output_data) {
      bool found = false;
      for (const Point &input_point : input_data_) {
        if (std::abs(point.x - input_point.x) < kEpsilon && std::abs(point.y - input_point.y) < kEpsilon) {
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
    return input_data_;
  }

 private:
  InType input_data_;
};

const std::array<TestType, 16> kTestParam = {
    CreateTestData(std::vector<Point>{Point{0.0, 0.0}, Point{1.0, 0.0}, Point{0.0, 1.0}},
                   std::vector<Point>{Point{0.0, 0.0}, Point{1.0, 0.0}, Point{0.0, 1.0}}),

    CreateTestData(std::vector<Point>{Point{0.0, 0.0}, Point{1.0, 0.0}, Point{1.0, 1.0}, Point{0.0, 1.0}},
                   std::vector<Point>{Point{0.0, 0.0}, Point{1.0, 0.0}, Point{1.0, 1.0}, Point{0.0, 1.0}}),

    CreateTestData(
        std::vector<Point>{Point{0.0, 0.0}, Point{2.0, 0.0}, Point{2.0, 2.0}, Point{0.0, 2.0}, Point{1.0, 1.0}},
        std::vector<Point>{Point{0.0, 0.0}, Point{2.0, 0.0}, Point{2.0, 2.0}, Point{0.0, 2.0}}),

    CreateTestData(std::vector<Point>{Point{0.0, 0.0}, Point{3.0, 0.0}, Point{3.0, 3.0}, Point{0.0, 3.0},
                                      Point{1.0, 1.0}, Point{2.0, 1.0}, Point{1.0, 2.0}, Point{2.0, 2.0}},
                   std::vector<Point>{Point{0.0, 0.0}, Point{3.0, 0.0}, Point{3.0, 3.0}, Point{0.0, 3.0}}),

    CreateTestData(std::vector<Point>{Point{0.0, 0.0}, Point{1.0, 0.0}, Point{2.0, 0.0}, Point{3.0, 0.0}},
                   std::vector<Point>{Point{0.0, 0.0}, Point{3.0, 0.0}}),

    CreateTestData(std::vector<Point>{Point{0.0, 0.0}, Point{0.0, 1.0}, Point{0.0, 2.0}, Point{0.0, 3.0}},
                   std::vector<Point>{Point{0.0, 0.0}, Point{0.0, 3.0}}),

    CreateTestData(
        std::vector<Point>{Point{0.0, 0.0}, Point{4.0, 0.0}, Point{2.0, 2.0}, Point{1.0, 1.0}, Point{3.0, 1.0},
                           Point{0.0, 4.0}, Point{4.0, 4.0}, Point{2.0, 5.0}},
        std::vector<Point>{Point{0.0, 0.0}, Point{4.0, 0.0}, Point{4.0, 4.0}, Point{2.0, 5.0}, Point{0.0, 4.0}}),

    CreateTestData(std::vector<Point>{Point{1.0, 1.0}, Point{1.0, 1.0}, Point{1.0, 1.0}},
                   std::vector<Point>{Point{1.0, 1.0}}),

    CreateTestData(
        std::vector<Point>{Point{0.0, 0.0}, Point{0.5, 0.0}, Point{1.0, 0.0}, Point{1.5, 0.0}, Point{2.0, 0.0},
                           Point{2.0, 0.5}, Point{2.0, 1.0}, Point{2.0, 1.5}, Point{2.0, 2.0}, Point{1.5, 2.0},
                           Point{1.0, 2.0}, Point{0.5, 2.0}, Point{0.0, 2.0}, Point{0.0, 1.5}, Point{0.0, 1.0},
                           Point{0.0, 0.5}, Point{0.5, 0.5}, Point{1.5, 0.5}, Point{0.5, 1.5}, Point{1.5, 1.5}},
        std::vector<Point>{Point{0.0, 0.0}, Point{2.0, 0.0}, Point{2.0, 2.0}, Point{0.0, 2.0}}),

    CreateTestData(
        std::vector<Point>{
            Point{3.0, 0.0},  Point{2.85, 0.93},   Point{2.43, 1.77},   Point{1.77, 2.43},   Point{0.93, 2.85},
            Point{0.0, 3.0},  Point{-0.93, 2.85},  Point{-1.77, 2.43},  Point{-2.43, 1.77},  Point{-2.85, 0.93},
            Point{-3.0, 0.0}, Point{-2.85, -0.93}, Point{-2.43, -1.77}, Point{-1.77, -2.43}, Point{-0.93, -2.85},
            Point{0.0, -3.0}, Point{0.93, -2.85},  Point{1.77, -2.43},  Point{2.43, -1.77},  Point{2.85, -0.93},
            Point{0.0, 0.0},  Point{1.0, 1.0},     Point{-1.0, 1.0},    Point{1.0, -1.0},    Point{-1.0, -1.0}},
        std::vector<Point>{Point{3.0, 0.0},     Point{2.85, 0.93},   Point{2.43, 1.77},   Point{1.77, 2.43},
                           Point{0.93, 2.85},   Point{0.0, 3.0},     Point{-0.93, 2.85},  Point{-1.77, 2.43},
                           Point{-2.43, 1.77},  Point{-2.85, 0.93},  Point{-3.0, 0.0},    Point{-2.85, -0.93},
                           Point{-2.43, -1.77}, Point{-1.77, -2.43}, Point{-0.93, -2.85}, Point{0.0, -3.0},
                           Point{0.93, -2.85},  Point{1.77, -2.43},  Point{2.43, -1.77},  Point{2.85, -0.93}}),
    CreateTestData(
        std::vector<Point>{Point{0.0, 4.0},   Point{1.2, 1.2},  Point{4.0, 0.0},   Point{1.2, -1.2},  Point{0.0, -4.0},
                           Point{-1.2, -1.2}, Point{-4.0, 0.0}, Point{-1.2, 1.2},  Point{2.5, 2.5},   Point{2.5, -2.5},
                           Point{-2.5, -2.5}, Point{-2.5, 2.5}, Point{0.0, 2.0},   Point{1.0, 1.0},   Point{2.0, 0.0},
                           Point{1.0, -1.0},  Point{0.0, -2.0}, Point{-1.0, -1.0}, Point{-2.0, 0.0},  Point{-1.0, 1.0},
                           Point{0.5, 0.5},   Point{1.5, 0.5},  Point{0.5, 1.5},   Point{-0.5, 0.5},  Point{-1.5, 0.5},
                           Point{0.5, -0.5},  Point{1.5, -0.5}, Point{-0.5, -0.5}, Point{-1.5, -0.5}, Point{0.0, 0.0}},
        std::vector<Point>{Point{0.0, 4.0}, Point{1.2, 1.2}, Point{2.5, 2.5}, Point{4.0, 0.0}, Point{2.5, -2.5},
                           Point{1.2, -1.2}, Point{0.0, -4.0}, Point{-1.2, -1.2}, Point{-2.5, -2.5}, Point{-4.0, 0.0},
                           Point{-2.5, 2.5}, Point{-1.2, 1.2}}),

    CreateTestData(std::vector<Point>{Point{5.0, 5.0}, Point{10.0, 10.0}, Point{0.0, 10.0}, Point{5.0, 15.0}},
                   std::vector<Point>{Point{5.0, 5.0}, Point{10.0, 10.0}, Point{5.0, 15.0}, Point{0.0, 10.0}}),
    CreateTestData(std::vector<Point>{Point{0.0, 0.0}, Point{3.0, 0.0}, Point{3.0, 3.0}, Point{0.0, 3.0},
                                      Point{1.0, 1.0}, Point{2.0, 2.0}},
                   std::vector<Point>{Point{0.0, 0.0}, Point{3.0, 0.0}, Point{3.0, 3.0}, Point{0.0, 3.0}}),
    CreateTestData(std::vector<Point>{Point{0.0, 0.0}, Point{4.0, 0.0}, Point{4.0, 4.0}, Point{0.0, 4.0},
                                      Point{2.0, 2.0}, Point{1.0, 3.0}, Point{3.0, 1.0}},
                   std::vector<Point>{Point{0.0, 0.0}, Point{4.0, 0.0}, Point{4.0, 4.0}, Point{0.0, 4.0}}),
    CreateTestData(std::vector<Point>{Point{0.0, 0.0}, Point{2.0, 0.0}, Point{4.0, 0.0}, Point{2.0, 2.0},
                                      Point{4.0, 4.0}, Point{0.0, 4.0}},
                   std::vector<Point>{Point{0.0, 0.0}, Point{4.0, 0.0}, Point{4.0, 4.0}, Point{0.0, 4.0}}),
    CreateTestData(
        std::vector<Point>{Point{1.0, 1.0}, Point{2.0, 2.0}, Point{3.0, 3.0}, Point{4.0, 4.0}, Point{5.0, 5.0}},
        std::vector<Point>{Point{1.0, 1.0}, Point{5.0, 5.0}})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEGrahamConvexHullMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_graham_cov_hall_constr),
                                           ppc::util::AddFuncTask<LuchnikovEGrahamConvexHullSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_graham_cov_hall_constr));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = LuchnikovEGrahamConvexHullFuncTests::PrintFuncTestName<LuchnikovEGrahamConvexHullFuncTests>;

INSTANTIATE_TEST_SUITE_P(GrahamConvexHullFuncTests, LuchnikovEGrahamConvexHullFuncTests, kGtestValues, kTestName);

TEST_P(LuchnikovEGrahamConvexHullFuncTests, RunFuncTests) {
  ExecuteTest(GetParam());
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LuchnikovEGrahamConvexHullFuncTests);

}  // namespace luchnikov_e_graham_cov_hall_constr
