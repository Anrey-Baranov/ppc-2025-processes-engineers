#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

#include "dorogin_v_contrast_enhancement/common/include/common.hpp"
#include "dorogin_v_contrast_enhancement/mpi/include/ops_mpi.hpp"
#include "dorogin_v_contrast_enhancement/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dorogin_v_contrast_enhancement {

class DoroginVContrastEnhancementFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int case_id = std::get<0>(params);

    input_data_ = {};
    expected_output_ = {};

    switch (case_id) {
      case 0: {
        input_data_ = {50, 100, 150, 200};
        expected_output_ = {0, 85, 170, 255};
        break;
      }
      case 1: {
        input_data_ = {0, 255};
        expected_output_ = {0, 255};
        break;
      }
      case 2: {
        input_data_ = {100, 100, 100, 100};
        expected_output_ = {100, 100, 100, 100};
        break;
      }
      case 3: {
        input_data_ = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        expected_output_.resize(10);
        for (int i = 0; i < 10; ++i) {
          expected_output_[i] = static_cast<uint8_t>(((input_data_[i] - 10) * 255) / 90);
        }
        break;
      }
      case 4: {
        input_data_ = {128};
        expected_output_ = {128};
        break;
      }
      case 5: {
        input_data_ = {0, 0, 255, 255};
        expected_output_ = {0, 0, 255, 255};
        break;
      }
      case 6: {
        input_data_.resize(1000);
        for (int i = 0; i < 1000; ++i) {
          input_data_[i] = static_cast<uint8_t>(i % 256);
        }
        expected_output_.resize(1000);
        for (int i = 0; i < 1000; ++i) {
          expected_output_[i] = static_cast<uint8_t>(input_data_[i]);
        }
        break;
      }
      case 7: {
        input_data_ = {50, 51, 52, 53, 54, 55};
        expected_output_.resize(6);
        for (int i = 0; i < 6; ++i) {
          expected_output_[i] = static_cast<uint8_t>(((input_data_[i] - 50) * 255) / 5);
        }
        break;
      }
      case 8: {
        input_data_ = {0, 1};
        expected_output_ = {0, 255};
        break;
      }
      case 9: {
        input_data_ = {254, 255};
        expected_output_ = {0, 255};
        break;
      }
      default:
        input_data_ = {0, 128, 255};
        expected_output_ = {0, 128, 255};
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(DoroginVContrastEnhancementFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(0, "basic_gradient"),      std::make_tuple(1, "min_max_only"),
    std::make_tuple(2, "uniform_values"),      std::make_tuple(3, "linear_sequence"),
    std::make_tuple(4, "single_pixel"),        std::make_tuple(5, "boundary_values"),
    std::make_tuple(6, "large_image"),         std::make_tuple(7, "small_range"),
    std::make_tuple(8, "adjacent_min_values"), std::make_tuple(9, "adjacent_max_values")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<DoroginVContrastEnhancementMPI, InType>(
                                               kTestParam, PPC_SETTINGS_dorogin_v_contrast_enhancement),
                                           ppc::util::AddFuncTask<DoroginVContrastEnhancementSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_dorogin_v_contrast_enhancement));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    DoroginVContrastEnhancementFuncTests::PrintFuncTestName<DoroginVContrastEnhancementFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, DoroginVContrastEnhancementFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace dorogin_v_contrast_enhancement
