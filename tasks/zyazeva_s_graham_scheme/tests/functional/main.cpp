#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zyazeva_s_graham_scheme/common/include/common.hpp"
#include "zyazeva_s_graham_scheme/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_graham_scheme/seq/include/ops_seq.hpp"

namespace zyazeva_s_graham_scheme {

namespace {

bool SamePoint(const Point &a, const Point &b) {
  return a.x == b.x && a.y == b.y;
}

bool PointLess(const Point &a, const Point &b) {
  return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

bool CompareHulls(std::vector<Point> actual, std::vector<Point> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }

  std::ranges::sort(actual.begin(), actual.end(), PointLess);
  std::ranges::sort(expected.begin(), expected.end(), PointLess);

  for (size_t i = 0; i < actual.size(); ++i) {
    if (!SamePoint(actual[i], expected[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

class ZyazevaGrahamRunFuncTestsSEQ : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static auto PrintTestParam(const TestType &param) -> std::string {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int64_t test_case = std::get<0>(params);

    switch (test_case) {
      case 0:  // triangle
        input_ = {{0, 0}, {1, 0}, {0, 1}};
        expected_ = {{0, 0}, {1, 0}, {0, 1}};
        break;

      case 1:  // square with inner point
        input_ = {{0, 0}, {0, 2}, {2, 2}, {2, 0}, {1, 1}};
        expected_ = {{0, 0}, {0, 2}, {2, 2}, {2, 0}};
        break;

      case 2:  // collinear
        input_ = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
        expected_ = {{0, 0}, {3, 3}};
        break;

      case 3:  // pentagon
        input_ = {{0, 1}, {1, 2}, {2, 1}, {1, 0}, {0, 0}};
        expected_ = {{0, 0}, {0, 1}, {1, 2}, {2, 1}, {1, 0}};
        break;

      case 4:  // min valid
        input_ = {{0, 0}, {1, 1}, {2, 0}};
        expected_ = {{0, 0}, {2, 0}, {1, 1}};
        break;

      case 5:  // one point
        input_ = {{0, 0}};
        expected_ = {};
        break;

      case 6:  // two points
        input_ = {{0, 0}, {1, 1}};
        expected_ = {};
        break;
      case 7:  // concave
        input_ = {{0, 0}, {2, 1}, {4, 0}, {3, 2}, {4, 4}, {2, 3}, {0, 4}, {1, 2}};
        expected_ = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
        break;
      default:
        input_ = {{0, 2}};
        expected_ = {};
        break;
    }
  }

  bool CheckTestOutputData(OutType &output) final {  // NOLINT
    return CompareHulls(output, expected_);
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
  OutType expected_;
};

class ZyazevaGrahamRunFuncTestsMPI : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static auto PrintTestParam(const TestType &param) -> std::string {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int64_t test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_ = {{0, 0}, {1, 0}, {0, 1}};
        expected_ = {{0, 0}, {1, 0}, {0, 1}};
        break;

      case 1:
        input_ = {{0, 0}, {0, 2}, {2, 2}, {2, 0}, {1, 1}};
        expected_ = {{0, 0}, {0, 2}, {2, 2}, {2, 0}};
        break;

      case 2:
        input_ = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
        expected_ = {{0, 0}, {3, 3}};
        break;

      case 3:
        input_ = {{0, 1}, {1, 2}, {2, 1}, {1, 0}, {0, 0}};
        expected_ = {{0, 0}, {0, 1}, {1, 2}, {2, 1}, {1, 0}};
        break;

      case 4:
        input_ = {{0, 0}, {1, 1}, {2, 0}};
        expected_ = {{0, 0}, {2, 0}, {1, 1}};
        break;

      case 5:
        input_ = {{0, 0}};
        expected_ = {};
        break;

      case 6:
        input_ = {{0, 0}, {1, 1}};
        expected_ = {};
        break;
      case 7:
        input_ = {{0, 0}, {2, 1}, {4, 0}, {3, 2}, {4, 4}, {2, 3}, {0, 4}, {1, 2}};
        expected_ = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
        break;
      default:
        input_ = {{0, 2}};
        expected_ = {};
        break;
    }
  }

  bool CheckTestOutputData(OutType &output) final {  // NOLINT
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      return CompareHulls(output, expected_);
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
  OutType expected_;
};
namespace {

TEST_P(ZyazevaGrahamRunFuncTestsSEQ, GrahamScanSEQ) {
  ExecuteTest(GetParam());
}

TEST_P(ZyazevaGrahamRunFuncTestsMPI, GrahamScanMPI) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTests = {std::make_tuple(0, "triangle"),   std::make_tuple(1, "square_with_inner"),
                                        std::make_tuple(2, "collinear"),  std::make_tuple(3, "pentagon"),
                                        std::make_tuple(4, "min_valid"),  std::make_tuple(5, "one_point"),
                                        std::make_tuple(6, "two_points"), std::make_tuple(7, "concave")};

const auto kSeqTasks =
    ppc::util::AddFuncTask<ZyazevaSGrahamSchemeSEQ, InType>(kTests, PPC_SETTINGS_zyazeva_s_graham_scheme);

const auto kMpiTasks =
    ppc::util::AddFuncTask<ZyazevaSGrahamSchemeMPI, InType>(kTests, PPC_SETTINGS_zyazeva_s_graham_scheme);

INSTANTIATE_TEST_SUITE_P(GrahamSchemeTestsSEQ, ZyazevaGrahamRunFuncTestsSEQ, ppc::util::ExpandToValues(kSeqTasks),
                         ZyazevaGrahamRunFuncTestsSEQ::PrintFuncTestName<ZyazevaGrahamRunFuncTestsSEQ>);

INSTANTIATE_TEST_SUITE_P(GrahamSchemeTestsMPI, ZyazevaGrahamRunFuncTestsMPI, ppc::util::ExpandToValues(kMpiTasks),
                         ZyazevaGrahamRunFuncTestsMPI::PrintFuncTestName<ZyazevaGrahamRunFuncTestsMPI>);

}  // namespace

}  // namespace zyazeva_s_graham_scheme
