#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cstddef>
#include <string>
#include <tuple>

#include "posternak_a_radix_merge_sort/common/include/common.hpp"
#include "posternak_a_radix_merge_sort/mpi/include/ops_mpi.hpp"
#include "posternak_a_radix_merge_sort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace posternak_a_radix_merge_sort {

class PosternakARadixMergeSortFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test = std::get<0>(params);

    switch (test) {
      case 1:
        input_data_ = {170, 45, 75, 90, 802, 24, 2, 66};
        expected_output_ = {2, 24, 45, 66, 75, 90, 170, 802};
        break;
      case 2:
        input_data_ = {1000, 100, 10, 1};
        expected_output_ = {1, 10, 100, 1000};
        break;
      case 3:
        input_data_ = {5, 4, 3, 2, 1};
        expected_output_ = {1, 2, 3, 4, 5};
        break;
      case 4:
        input_data_ = {1, 2, 3, 4, 5};
        expected_output_ = {1, 2, 3, 4, 5};
        break;
      case 5:
        input_data_ = {-42, -43, 42};
        expected_output_ = {-43, -42, 42};
        break;
      case 6:
        input_data_ = {-5, -1, -3, -2, -4};
        expected_output_ = {-5, -4, -3, -2, -1};
        break;
      case 7:
        input_data_ = {-10, 10, 0, -5, 5};
        expected_output_ = {-10, -5, 0, 5, 10};
        break;
      case 8:
        input_data_ = {999, -999, 0, 100, -100};
        expected_output_ = {-999, -100, 0, 100, 999};
        break;
      case 9:
        input_data_ = {0, 0, 0, 0};
        expected_output_ = {0, 0, 0, 0};
        break;
      case 10:
        input_data_ = {123456789, 987654321, 111111111};
        expected_output_ = {111111111, 123456789, 987654321};
        break;
      case 11:
        input_data_ = {1, 22, 333, 4444, 55555};
        expected_output_ = {1, 22, 333, 4444, 55555};
        break;
      case 12:
        input_data_ = {22, 1, 333, 55555, 4444};
        expected_output_ = {1, 22, 333, 4444, 55555};
        break;
      case 13:
        input_data_ = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        expected_output_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        break;
      case 14:
        input_data_ = {-1, -2, -3, 1, 2, 3};
        expected_output_ = {-3, -2, -1, 1, 2, 3};
        break;
      case 15:
        input_data_ = {100, 200, 300, 400, 500};
        expected_output_ = {100, 200, 300, 400, 500};
        break;
      // Граничные
      case 16:
        input_data_ = {1};
        expected_output_ = {1};
        break;
      case 17:
        input_data_ = {-1};
        expected_output_ = {-1};
        break;
      case 18:
        input_data_ = {INT_MAX, INT_MIN, 0};
        expected_output_ = {INT_MIN, 0, INT_MAX};
        break;
      default:
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(PosternakARadixMergeSortFuncTests, RadixMergeSort) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 18> kTestParam = {std::make_tuple(1, "mixed_positive"),
                                             std::make_tuple(2, "powers_of_10"),
                                             std::make_tuple(3, "reverse_order"),
                                             std::make_tuple(4, "already_sorted"),
                                             std::make_tuple(5, "correct_sort"),
                                             std::make_tuple(6, "all_negative"),
                                             std::make_tuple(7, "mixed_sign"),
                                             std::make_tuple(8, "big_range"),
                                             std::make_tuple(9, "all_zeros"),
                                             std::make_tuple(10, "big_numbers"),
                                             std::make_tuple(11, "sorted_scale_radix"),
                                             std::make_tuple(12, "unsorted_scale_radix"),
                                             std::make_tuple(13, "ten_elements"),
                                             std::make_tuple(14, "symmetric"),
                                             std::make_tuple(15, "only_hundred_radix"),
                                             std::make_tuple(16, "one_positive_number"),
                                             std::make_tuple(17, "one_negative_number"),
                                             std::make_tuple(18, "limits_numbers")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PosternakARadixMergeSortMPI, InType>(kTestParam, PPC_SETTINGS_posternak_a_radix_merge_sort),
    ppc::util::AddFuncTask<PosternakARadixMergeSortSEQ, InType>(kTestParam, PPC_SETTINGS_posternak_a_radix_merge_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PosternakARadixMergeSortFuncTests::PrintFuncTestName<PosternakARadixMergeSortFuncTests>;

INSTANTIATE_TEST_SUITE_P(IntTests, PosternakARadixMergeSortFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace posternak_a_radix_merge_sort
