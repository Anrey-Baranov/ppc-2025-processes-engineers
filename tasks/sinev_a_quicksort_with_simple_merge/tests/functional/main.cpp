#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "sinev_a_quicksort_with_simple_merge/common/include/common.hpp"
#include "sinev_a_quicksort_with_simple_merge/mpi/include/ops_mpi.hpp"
#include "sinev_a_quicksort_with_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sinev_a_quicksort_with_simple_merge {

class SinevAQuicksortWithSimpleMergeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {5, 3, 8, 1, 9, 2};
        break;

      case 1:
        input_data_ = {10, -5, 7, 0, -15, 3};
        break;

      case 2:
        input_data_ = {1, 2, 3, 4, 5, 6};
        break;

      case 3:
        input_data_ = {6, 5, 4, 3, 2, 1};
        break;

      case 4:
        input_data_ = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
        break;

      case 5:
        input_data_ = {7, 7, 7, 7, 7};
        break;

      case 6:
        input_data_ = {42};
        break;

      case 7:
        input_data_ = {1, 2};
        break;

      case 8:
        input_data_ = {2, 1};
        break;

      case 9:
        input_data_ = {9, 3, 7, 1, 8, 2, 6, 4, 5, 0};
        break;

      case 10:
        input_data_ = {11, 3, 9, 1, 7, 5, 8, 2, 6, 4, 10};
        break;

      case 11:
        input_data_ = {0, -1, 0, 1, -2, 2};
        break;

      case 12:
        input_data_ = {2147483647, -2147483648, 0, 100, -100};
        break;

      default:
        input_data_ = {100, -50, 0, 25, -25, 75, -75, 50, -100};
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (input_data_.empty() && !output_data.empty()) {
      return false;
    }

    if (!input_data_.empty() && output_data.empty()) {
      return false;
    }

    if (output_data.size() != input_data_.size()) {
      return false;
    }

    for (size_t i = 1; i < output_data.size(); i++) {
      if (output_data[i] < output_data[i - 1]) {
        return false;
      }
    }

    std::vector<int> sorted_input = input_data_;
    std::vector<int> sorted_output = output_data;

    std::ranges::sort(sorted_input);
    std::ranges::sort(sorted_output);

    return sorted_input == sorted_output;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SinevAQuicksortWithSimpleMergeFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {
    std::make_tuple(0, "basic_case"),      std::make_tuple(1, "with_negatives"),  std::make_tuple(2, "already_sorted"),
    std::make_tuple(3, "reverse_sorted"),  std::make_tuple(4, "with_duplicates"), std::make_tuple(5, "all_equal"),
    std::make_tuple(6, "single_element"),  std::make_tuple(7, "two_sorted"),      std::make_tuple(8, "two_unsorted"),
    std::make_tuple(9, "even_size_large"), std::make_tuple(10, "odd_size_large"), std::make_tuple(11, "with_zeros"),
    std::make_tuple(12, "extreme_values"), std::make_tuple(13, "mixed_case")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SinevAQuicksortWithSimpleMergeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_sinev_a_quicksort_with_simple_merge),
                                           ppc::util::AddFuncTask<SinevAQuicksortWithSimpleMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_sinev_a_quicksort_with_simple_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    SinevAQuicksortWithSimpleMergeFuncTests::PrintFuncTestName<SinevAQuicksortWithSimpleMergeFuncTests>;

INSTANTIATE_TEST_SUITE_P(QuickSortTests, SinevAQuicksortWithSimpleMergeFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sinev_a_quicksort_with_simple_merge
