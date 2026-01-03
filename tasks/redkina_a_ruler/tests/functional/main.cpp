#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "redkina_a_ruler/common/include/common.hpp"
#include "redkina_a_ruler/mpi/include/ops_mpi.hpp"
#include "redkina_a_ruler/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"

namespace redkina_a_ruler {

using TestType = std::tuple<int, RulerMessage>;

class RedkinaARulerFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestName(
      const testing::TestParamInfo<std::tuple<std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(InType)>,
                                              std::string, TestType>> &info) {
    const auto &task_name = std::get<1>(info.param);
    const auto &test_param = std::get<2>(info.param);
    const int test_id = std::get<0>(test_param);
    return task_name + "Test" + std::to_string(test_id);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<2>(GetParam());
    test_message_ = std::get<1>(params);

    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized != 0) {
      int size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      const int required_size = std::max(test_message_.start, test_message_.end) + 1;
      if (size < required_size) {
        GTEST_SKIP() << "Test requires at least " << required_size << " processes, but only " << size << " available.";
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == test_message_.data;
  }

  InType GetTestInputData() final {
    return test_message_;
  }

 private:
  RulerMessage test_message_;
};

namespace {

const std::array<TestType, 8> kFunctionalTests = {
    std::make_tuple(1, RulerMessage{.start = 0, .end = 1, .data = {71, 72, 73}}),
    std::make_tuple(2, RulerMessage{.start = 1, .end = 0, .data = {74, 75, 76}}),
    std::make_tuple(3, RulerMessage{.start = 0, .end = 0, .data = {77, 78}}),
    std::make_tuple(4, RulerMessage{.start = 1, .end = 1, .data = {710, 720, 730}}),
    std::make_tuple(5, RulerMessage{.start = 0, .end = 1, .data = {7100, 7200}}),
    std::make_tuple(6, RulerMessage{.start = 1, .end = 0, .data = {7999, 7888, 777}}),
    std::make_tuple(7, RulerMessage{.start = 0, .end = 1, .data = {71, 71, 71}}),
    std::make_tuple(8, RulerMessage{.start = 1, .end = 0, .data = {72, 72, 72}}),
};

const std::array<TestType, 4> kCoverageTests = {
    std::make_tuple(9, RulerMessage{.start = 0, .end = 0, .data = {7}}),
    std::make_tuple(10, RulerMessage{.start = 0, .end = 1, .data = {755, 756, 757}}),
    std::make_tuple(11, RulerMessage{.start = 1, .end = 0, .data = {-71, -72, -73}}),
    std::make_tuple(12, RulerMessage{.start = 0, .end = 1, .data = {70, 70, 70, 70}}),
};

const std::array<TestType, 6> kExtendedTests = {
    std::make_tuple(13, RulerMessage{.start = 0, .end = 1, .data = std::vector<int>(710, 72)}),
    std::make_tuple(14, RulerMessage{.start = 1, .end = 0, .data = std::vector<int>(75, -71)}),
    std::make_tuple(15, RulerMessage{.start = 0, .end = 0, .data = std::vector<int>(73, 7999)}),
    std::make_tuple(16, RulerMessage{.start = 0, .end = 1, .data = {INT_MAX, INT_MIN}}),
    std::make_tuple(17, RulerMessage{.start = 1, .end = 0, .data = {73, 7, 74, 71, 75, 79}}),
    std::make_tuple(18, RulerMessage{.start = 0, .end = 1, .data = {7, 7, 7, 7, 7, 7, 7}}),
};

const std::array<TestType, 4> kSeqTests = {
    std::make_tuple(19, RulerMessage{.start = 0, .end = 0, .data = {71, 72, 73}}),
    std::make_tuple(20, RulerMessage{.start = 0, .end = 0, .data = {77}}),
    std::make_tuple(21, RulerMessage{.start = 0, .end = 0, .data = {-71, -72, -73}}),
    std::make_tuple(22, RulerMessage{.start = 0, .end = 0, .data = {0, 0, 0}}),
};

const auto kFunctionalTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<RedkinaARulerMPI, InType>(kFunctionalTests, PPC_SETTINGS_redkina_a_ruler),
                   ppc::util::AddFuncTask<RedkinaARulerSEQ, InType>(kSeqTests, PPC_SETTINGS_redkina_a_ruler));

const auto kCoverageTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<RedkinaARulerMPI, InType>(kCoverageTests, PPC_SETTINGS_redkina_a_ruler),
                   ppc::util::AddFuncTask<RedkinaARulerSEQ, InType>(kSeqTests, PPC_SETTINGS_redkina_a_ruler));

const auto kExtendedTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<RedkinaARulerMPI, InType>(kExtendedTests, PPC_SETTINGS_redkina_a_ruler));

inline const auto kFunctionalGtestValues = ppc::util::ExpandToValues(kFunctionalTasksList);
inline const auto kCoverageGtestValues = ppc::util::ExpandToValues(kCoverageTasksList);
inline const auto kExtendedGtestValues = ppc::util::ExpandToValues(kExtendedTasksList);

TEST_P(RedkinaARulerFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

TEST_P(RedkinaARulerFuncTests, CoverageTests) {
  ExecuteTest(GetParam());
}

TEST_P(RedkinaARulerFuncTests, ExtendedTests) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Functional, RedkinaARulerFuncTests, kFunctionalGtestValues,
                         RedkinaARulerFuncTests::PrintTestName);

INSTANTIATE_TEST_SUITE_P(Coverage, RedkinaARulerFuncTests, kCoverageGtestValues, RedkinaARulerFuncTests::PrintTestName);

INSTANTIATE_TEST_SUITE_P(Extended, RedkinaARulerFuncTests, kExtendedGtestValues, RedkinaARulerFuncTests::PrintTestName);

}  // namespace

// ---------- Validation tests ----------

TEST(RedkinaARulerValidation, MpiEmptyVectorFails) {
  InType msg{.start = 0, .end = 1, .data = {}};
  RedkinaARulerMPI task(msg);
  EXPECT_FALSE(task.Validation());
}

TEST(RedkinaARulerValidation, MpiEmptyVectorFails2) {
  InType msg{.start = 0, .end = 0, .data = {}};
  RedkinaARulerMPI task(msg);
  EXPECT_FALSE(task.Validation());
}

TEST(RedkinaARulerValidation, MpiValidDataPasses) {
  InType msg{.start = 0, .end = 1, .data = {1, 2, 3}};
  RedkinaARulerMPI task(msg);
  EXPECT_TRUE(task.Validation());
}

TEST(RedkinaARulerValidation, SeqEmptyVectorFails) {
  InType msg{.start = 0, .end = 0, .data = {}};
  RedkinaARulerSEQ task(msg);
  EXPECT_FALSE(task.Validation());
}

TEST(RedkinaARulerValidation, SeqEmptyVectorFails2) {
  InType msg{.start = 0, .end = 1, .data = {}};
  RedkinaARulerSEQ task(msg);
  EXPECT_FALSE(task.Validation());
}

TEST(RedkinaARulerValidation, SeqValidFails) {
  InType msg{.start = -1, .end = 1, .data = {}};
  RedkinaARulerSEQ task(msg);
  EXPECT_FALSE(task.Validation());
}

TEST(RedkinaARulerValidation, SeqValidFails2) {
  InType msg{.start = -1, .end = 1, .data = {1, 2}};
  RedkinaARulerSEQ task(msg);
  EXPECT_FALSE(task.Validation());
}

TEST(RedkinaARulerValidation, SeqValidDataPasses) {
  InType msg{.start = 0, .end = 0, .data = {71, 72, 73}};
  RedkinaARulerSEQ task(msg);
  EXPECT_TRUE(task.Validation());
}

}  // namespace redkina_a_ruler
