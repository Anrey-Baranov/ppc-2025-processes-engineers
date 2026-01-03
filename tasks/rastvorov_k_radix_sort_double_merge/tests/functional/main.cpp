#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "rastvorov_k_radix_sort_double_merge/common/include/common.hpp"
#include "rastvorov_k_radix_sort_double_merge/mpi/include/ops_mpi.hpp"
#include "rastvorov_k_radix_sort_double_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace rastvorov_k_radix_sort_double_merge {

class RastvorovKRadixSortDoubleMergeRunFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);

    expected_ = input_data_;
    std::ranges::sort(expected_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(RastvorovKRadixSortDoubleMergeRunFuncTestsProcesses, RadixSortDoubleBasicCases) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(std::vector<double>{}, "empty"),
    std::make_tuple(std::vector<double>{42.0}, "single"),
    std::make_tuple(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}, "already_sorted"),
    std::make_tuple(std::vector<double>{5.0, 4.0, 3.0, 2.0, 1.0}, "reverse_sorted"),
    std::make_tuple(std::vector<double>{1.0, 1.0, 2.0, 2.0, 0.0, 0.0}, "duplicates"),
    std::make_tuple(std::vector<double>{10.5, -1.25, 7.0, 7.0, 2.0, -100.0, 50.0, 3.14}, "mixed_values"),
    std::make_tuple(std::vector<double>{0.0, -0.0, 0.0, -0.0}, "signed_zero"),
    std::make_tuple(std::vector<double>{-1.0e308, 1.0e308, -1.0e-308, 1.0e-308}, "extremes"),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RastvorovKRadixSortDoubleMergeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_rastvorov_k_radix_sort_double_merge),
                                           ppc::util::AddFuncTask<RastvorovKRadixSortDoubleMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_rastvorov_k_radix_sort_double_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = RastvorovKRadixSortDoubleMergeRunFuncTestsProcesses::PrintFuncTestName<
    RastvorovKRadixSortDoubleMergeRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(RastvorovKRadixSortDoubleMergeFuncTests, RastvorovKRadixSortDoubleMergeRunFuncTestsProcesses,
                         kGtestValues, kFuncTestName);

}  // namespace

}  // namespace rastvorov_k_radix_sort_double_merge
