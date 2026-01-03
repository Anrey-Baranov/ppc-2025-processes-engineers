#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "dorogin_v_min_vector_value/common/include/common.hpp"
#include "dorogin_v_min_vector_value/mpi/include/ops_mpi.hpp"
#include "dorogin_v_min_vector_value/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dorogin_v_min_vector_value {

class DoroginVMinVectorValueFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "case_" + std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    const TestType &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_vector_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto it = std::ranges::min_element(test_vector_);
    const int expected = (it != test_vector_.end()) ? *it : 0;
    return output_data == expected;
  }

  InType GetTestInputData() final {
    return test_vector_;
  }

 private:
  std::vector<int> test_vector_;
};

namespace {

TEST_P(DoroginVMinVectorValueFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

TEST_P(DoroginVMinVectorValueFuncTests, CoverageTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kFunctionalTests = {std::make_tuple(1, std::vector<int>{3, 1, 4, 1, 5, 9}),
                                                   std::make_tuple(2, std::vector<int>{10, -3, 7, 2, -1, 0}),
                                                   std::make_tuple(3, std::vector<int>{42}),
                                                   std::make_tuple(4, std::vector<int>{0, 0, 0, 0}),
                                                   std::make_tuple(5, std::vector<int>{-5, -4, -3, -2, -1}),
                                                   std::make_tuple(6, std::vector<int>{100, 50, 25, 12, 6, 3}),
                                                   std::make_tuple(7, std::vector<int>{7, 6, 5, 4, 3, 2, 1}),
                                                   std::make_tuple(8, std::vector<int>{-10, 5, -20, 15, 0}),
                                                   std::make_tuple(9, std::vector<int>{8, 6, 7, 5, 3, 0, 9}),
                                                   std::make_tuple(10, []() {
  std::vector<int> v(256);
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<int>(i % 17) - 8;
  }
  return v;
}())};

const std::array<TestType, 12> kCoverageTests = {std::make_tuple(11, std::vector<int>{1, 2, 3}),
                                                 std::make_tuple(12, std::vector<int>{-1, 2, -3, 4}),
                                                 std::make_tuple(13, std::vector<int>{5, 4, 3, 2, 1}),
                                                 std::make_tuple(14, std::vector<int>{0, 1, 2, 3, 4, 5}),
                                                 std::make_tuple(15, std::vector<int>{-100, 0, 100}),
                                                 std::make_tuple(16, std::vector<int>{9, 9, 9, 9, 9}),
                                                 std::make_tuple(17, std::vector<int>{-2, -2, -2, -2}),
                                                 std::make_tuple(18, std::vector<int>{50, 40, 30, 20, 10, 0}),
                                                 std::make_tuple(19, std::vector<int>{1, -1}),
                                                 std::make_tuple(20, std::vector<int>{7, 0, -7}),
                                                 std::make_tuple(21,
                                                                 []() {
  std::vector<int> v(128);
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<int>(128 - i);
  }
  return v;
}()),
                                                 std::make_tuple(22, []() {
  std::vector<int> v(64);
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<int>(i * i % 31) - 15;
  }
  return v;
}())};

const auto kFunctionalTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<dorogin_v_min_vector_value::DoroginVMinVectorValueMPI, InType>(
                       kFunctionalTests, PPC_SETTINGS_dorogin_v_min_vector_value),
                   ppc::util::AddFuncTask<dorogin_v_min_vector_value::DoroginVMinVectorValueSEQ, InType>(
                       kFunctionalTests, PPC_SETTINGS_dorogin_v_min_vector_value));

const auto kCoverageTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<dorogin_v_min_vector_value::DoroginVMinVectorValueMPI, InType>(
                       kCoverageTests, PPC_SETTINGS_dorogin_v_min_vector_value),
                   ppc::util::AddFuncTask<dorogin_v_min_vector_value::DoroginVMinVectorValueSEQ, InType>(
                       kCoverageTests, PPC_SETTINGS_dorogin_v_min_vector_value));

inline const auto kFunctionalGtestValues = ppc::util::ExpandToValues(kFunctionalTasksList);
inline const auto kCoverageGtestValues = ppc::util::ExpandToValues(kCoverageTasksList);
inline const auto kPerfTestName = DoroginVMinVectorValueFuncTests::PrintFuncTestName<DoroginVMinVectorValueFuncTests>;

INSTANTIATE_TEST_SUITE_P(Functional, DoroginVMinVectorValueFuncTests, kFunctionalGtestValues, kPerfTestName);
INSTANTIATE_TEST_SUITE_P(Coverage, DoroginVMinVectorValueFuncTests, kCoverageGtestValues, kPerfTestName);

TEST(DoroginVMinVectorValueValidation, MpiEmptyVectorValidationFails) {
  InType vec = {};
  DoroginVMinVectorValueMPI task(vec);
  EXPECT_FALSE(task.Validation());
}

TEST(DoroginVMinVectorValueValidation, SeqEmptyVectorValidationFails) {
  InType vec = {};
  DoroginVMinVectorValueSEQ task(vec);
  EXPECT_FALSE(task.Validation());
}

}  // namespace
}  // namespace dorogin_v_min_vector_value
