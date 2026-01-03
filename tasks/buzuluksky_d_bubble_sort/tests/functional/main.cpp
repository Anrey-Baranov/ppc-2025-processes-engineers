#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "buzuluksky_d_bubble_sort/common/include/common.hpp"
#include "buzuluksky_d_bubble_sort/mpi/include/ops_mpi.hpp"
#include "buzuluksky_d_bubble_sort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace buzuluksky_d_bubble_sort {

class BuzulukskyDBubbleSortFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output) final {
    auto expected = input_;
    std::ranges::sort(expected);
    return output == expected;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
};

namespace {

TEST_P(BuzulukskyDBubbleSortFuncTests, BubbleSortTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParams{
    TestType{std::vector<int>{}, "empty"},
    TestType{std::vector<int>{3}, "one_elem"},
    TestType{std::vector<int>{3, 5, 21, 1, 4}, "random_5"},
    TestType{std::vector<int>{1, 2, 3, 4, 5}, "sorted"},
    TestType{std::vector<int>{-1, 100, 0, -50, 50}, "negative"},
    TestType{std::vector<int>{1, 1, 1, 1}, "same_numbers"},
    TestType{std::vector<int>{1, 3, 2, 5, 8, 7, 4, 6, 9, 0}, "random_10"},
    TestType{std::vector<int>{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, "reverse"},
    TestType{std::vector<int>{1000, -1000, 500, -500, 250, -250, 125, -125}, "large_range"},
};

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<BuzulukskyDBubbleSortMPI, InType>(kTestParams, PPC_SETTINGS_buzuluksky_d_bubble_sort),
    ppc::util::AddFuncTask<BuzulukskyDBubbleSortSEQ, InType>(kTestParams, PPC_SETTINGS_buzuluksky_d_bubble_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTaskList);
const auto kTestName = BuzulukskyDBubbleSortFuncTests::PrintFuncTestName<BuzulukskyDBubbleSortFuncTests>;

INSTANTIATE_TEST_SUITE_P(BubbleSortTests, BuzulukskyDBubbleSortFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace buzuluksky_d_bubble_sort
