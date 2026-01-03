#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "posternak_a_increase_contrast/common/include/common.hpp"
#include "posternak_a_increase_contrast/mpi/include/ops_mpi.hpp"
#include "posternak_a_increase_contrast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace posternak_a_increase_contrast {

// Проверка допустимости округления
static bool VectorsAlmostEqual(const OutType &a, const OutType &b, unsigned char scale = 1) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])) > scale) {
      return false;
    }
  }
  return true;
}

class PosternakAIncreaseContrastFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
        input_data_ = {50, 100, 150, 200};
        expected_output_ = {0, 85, 170, 255};
        break;
      case 2:
        input_data_ = {10, 200, 50, 180, 30};
        expected_output_ = {0, 255, 54, 228, 27};
        break;
      case 3:
        input_data_ = {120, 130, 125, 128};
        expected_output_ = {0, 255, 128, 204};
        break;
      case 4:
        input_data_ = {70, 160, 90, 200, 110};
        expected_output_ = {0, 177, 40, 255, 79};
        break;
      case 5:
        input_data_ = {0, 64, 128, 192, 255};
        expected_output_ = {0, 64, 128, 192, 255};
        break;
      // Граничные случаи
      case 6:
        input_data_ = {100, 100, 100};
        expected_output_ = {128, 128, 128};
        break;
      case 7:
        input_data_ = {0, 255, 0, 255};
        expected_output_ = {0, 255, 0, 255};
        break;
      case 8:
        input_data_ = {0, 0, 0};
        expected_output_ = {128, 128, 128};
        break;
      case 9:
        input_data_ = {255, 255};
        expected_output_ = {128, 128};
        break;
      case 10:
        input_data_ = {255};
        expected_output_ = {128};
        break;
      default:
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (*std::ranges::min_element(input_data_) == *std::ranges::max_element(input_data_)) {
      return std::ranges::all_of(output_data, [](unsigned char v) { return v == 128; });
    }
    return VectorsAlmostEqual(output_data, expected_output_, 1);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(PosternakAIncreaseContrastFuncTests, IncreaseContrast) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(1, "linear"),
                                             std::make_tuple(2, "big_amplitude"),
                                             std::make_tuple(3, "middle_pixels"),
                                             std::make_tuple(4, "default"),
                                             std::make_tuple(5, "full_scale"),
                                             std::make_tuple(6, "one_color"),
                                             std::make_tuple(7, "black_white_image"),
                                             std::make_tuple(8, "black_image"),
                                             std::make_tuple(9, "white_image"),
                                             std::make_tuple(10, "one_pixel")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<PosternakAIncreaseContrastMPI, InType>(
                                               kTestParam, PPC_SETTINGS_posternak_a_increase_contrast),
                                           ppc::util::AddFuncTask<PosternakAIncreaseContrastSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_posternak_a_increase_contrast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PosternakAIncreaseContrastFuncTests::PrintFuncTestName<PosternakAIncreaseContrastFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorTests, PosternakAIncreaseContrastFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace posternak_a_increase_contrast
