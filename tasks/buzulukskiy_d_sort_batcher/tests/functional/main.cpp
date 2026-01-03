#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include "buzulukskiy_d_sort_batcher/common/include/common.hpp"
#include "buzulukskiy_d_sort_batcher/mpi/include/ops_mpi.hpp"
#include "buzulukskiy_d_sort_batcher/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace buzulukskiy_d_sort_batcher {

class BuzulukskiyDSortBatcherFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    const auto &params = GetParam();
    const auto &test_param = std::get<TestType>(params);
    input_data_ = std::get<0>(test_param);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output) final {
    auto expected_data = input_data_;
    std::ranges::sort(expected_data);
    return output == expected_data;
  }

 private:
  InType input_data_;
};

namespace {
const std::array<TestType, 10> kTestParams = {
    TestType{InType{}, "empty"},
    TestType{InType{5}, "single_element"},
    TestType{InType{3, 5, 21, 1, 4}, "random_5"},
    TestType{InType{1, 2, 3, 4, 5}, "already_sorted"},
    TestType{InType{-1, -5, 3, 0, 2}, "mixed_negative"},
    TestType{InType{1, 1, 1, 1}, "all_equal"},
    TestType{InType{1000, -1000, 500, -500, 250, -250}, "wide_range"},
    TestType{InType{9, 7, 5, 3, 1, 2, 4, 6, 8, 0}, "random_10"},
    TestType{InType{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, "reverse_sorted"},
    TestType{InType{4, -2, 7, 0, -2, 9, 1}, "uneven_distribution"},
};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<BuzulukskiyDSortBatcherSEQ, InType>(kTestParams, PPC_SETTINGS_buzulukskiy_d_sort_batcher),
    ppc::util::AddFuncTask<BuzulukskiyDSortBatcherMPI, InType>(kTestParams, PPC_SETTINGS_buzulukskiy_d_sort_batcher));

const auto kValues = ppc::util::ExpandToValues(kTasks);

TEST_P(BuzulukskiyDSortBatcherFuncTests, Run) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(SortBatcherFunctionalTests, BuzulukskiyDSortBatcherFuncTests, kValues);
}  // namespace
}  // namespace buzulukskiy_d_sort_batcher
