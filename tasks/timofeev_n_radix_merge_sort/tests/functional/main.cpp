#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "timofeev_n_radix_merge_sort/common/include/common.hpp"
#include "timofeev_n_radix_merge_sort/mpi/include/ops_mpi.hpp"
#include "timofeev_n_radix_merge_sort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace timofeev_n_radix_merge_sort {

class RadixMergeRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<3>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<2>(params);
    expected_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_data_.size()) {
      return false;
    }
    bool yesyes = true;
    for (size_t i = 0; i < output_data.size(); i++) {
      yesyes &= output_data[i] == expected_data_[i];
    }
    return yesyes;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_data_;
};

namespace {

TEST_P(RadixMergeRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

std::vector<int> sorted5 = {0, 1, 2, 3, 4};
std::vector<int> reverse5 = {4, 3, 2, 1, 0};
std::vector<int> zeros5(5);

std::vector<int> sorted4 = {0, 1, 2, 3};
std::vector<int> reverse4 = {3, 2, 1, 0};
std::vector<int> zeros4(4);

std::vector<int> sorted3 = {0, 1, 2};
std::vector<int> reverse3 = {2, 1, 0};
std::vector<int> zeros3(3);

std::vector<int> sorted2 = {0, 1};
std::vector<int> reverse2 = {1, 0};
std::vector<int> zeros2(2);

std::vector<int> sorted1 = {0};
std::vector<int> reverse1 = {0};
std::vector<int> zeros1(1);

std::vector<int> sorted_neg5 = {-4, -3, -2, -1, 0};
std::vector<int> reverse_neg5 = {0, -1, -2, -3, -4};

std::vector<int> sorted_neg4 = {-4, -3, -2, -1};
std::vector<int> reverse_neg4 = {-1, -2, -3, -4};

std::vector<int> sorted_neg3 = {-4, -3, -2};
std::vector<int> reverse_neg3 = {-2, -3, -4};

std::vector<int> sorted_neg2 = {-4, -3};
std::vector<int> reverse_neg2 = {-3, -4};

std::vector<int> sorted_neg1 = {-4};
std::vector<int> reverse_neg1 = {-4};

std::vector<int> sorted_thegrandfinal = {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
std::vector<int> reverse_thegrandfinal = {5, 4, 3, 2, 1, 0, -1, -2, -3, -4};
std::vector<int> zeros_thegrandfinal(10);

const std::array<TestType, 11> kTestParam = {
    std::make_tuple(sorted5, zeros5, reverse5, "first"),
    std::make_tuple(sorted4, zeros4, reverse4, "second"),
    std::make_tuple(sorted3, zeros3, reverse3, "third"),
    std::make_tuple(sorted2, zeros2, reverse2, "fourth"),
    std::make_tuple(sorted1, zeros1, reverse1, "fifth"),
    std::make_tuple(sorted_neg5, zeros5, reverse_neg5, "firstNeg"),
    std::make_tuple(sorted_neg4, zeros4, reverse_neg4, "secondNeg"),
    std::make_tuple(sorted_neg3, zeros3, reverse_neg3, "thirdNeg"),
    std::make_tuple(sorted_neg2, zeros2, reverse_neg2, "fourthNeg"),
    std::make_tuple(sorted_neg1, zeros1, reverse_neg1, "fifthNeg"),
    std::make_tuple(sorted_thegrandfinal, zeros_thegrandfinal, reverse_thegrandfinal, "THEGRANDFINAL")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TimofeevNRadixMergeMPI, InType>(kTestParam, PPC_SETTINGS_timofeev_n_radix_merge_sort),
    ppc::util::AddFuncTask<TimofeevNRadixMergeSEQ, InType>(kTestParam, PPC_SETTINGS_timofeev_n_radix_merge_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RadixMergeRunFuncTestsProcesses::PrintFuncTestName<RadixMergeRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, RadixMergeRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace timofeev_n_radix_merge_sort
