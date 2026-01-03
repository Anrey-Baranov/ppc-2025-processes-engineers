#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vlasova_a_image_smoothing/common/include/common.hpp"
#include "vlasova_a_image_smoothing/mpi/include/ops_mpi.hpp"
#include "vlasova_a_image_smoothing/seq/include/ops_seq.hpp"

namespace vlasova_a_image_smoothing {

class VlasovaARunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    window_size_ = std::get<0>(params);

    const int width = 64;
    const int height = 64;

    input_data_.width = width;
    input_data_.height = height;
    input_data_.data.resize(static_cast<std::size_t>(width) * height);

    for (int row_idx = 0; row_idx < height; ++row_idx) {
      for (int col_idx = 0; col_idx < width; ++col_idx) {
        const std::size_t index = (static_cast<std::size_t>(row_idx) * width) + col_idx;
        const auto gradient = static_cast<std::uint8_t>((col_idx + row_idx) * 2);
        input_data_.data[index] = gradient;
      }
    }

    for (std::size_t counter = 0; counter < input_data_.data.size() / 20; ++counter) {
      const std::size_t index = (counter * 97) % input_data_.data.size();
      if (counter % 3 == 0) {
        input_data_.data[index] = 0;
      } else if (counter % 3 == 1) {
        input_data_.data[index] = 255;
      } else {
        input_data_.data[index] = 128;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.width != input_data_.width || output_data.height != input_data_.height ||
        output_data.data.size() != input_data_.data.size()) {
      return false;
    }

    const std::uint8_t first = output_data.data[0];
    for (std::size_t idx = 1; idx < output_data.data.size(); ++idx) {
      if (output_data.data[idx] != first) {
        return true;
      }
    }

    return false;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  int window_size_ = 3;
  InType input_data_;
};

namespace {

TEST_P(VlasovaARunFuncTests, MedianFilterTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "window3"), std::make_tuple(5, "window5"),
                                            std::make_tuple(7, "window7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VlasovaAImageSmoothingMPI, InType>(kTestParam, PPC_SETTINGS_vlasova_a_image_smoothing),
    ppc::util::AddFuncTask<VlasovaAImageSmoothingSEQ, InType>(kTestParam, PPC_SETTINGS_vlasova_a_image_smoothing));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VlasovaARunFuncTests::PrintFuncTestName<VlasovaARunFuncTests>;

INSTANTIATE_TEST_SUITE_P(MedianFilterTests, VlasovaARunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace vlasova_a_image_smoothing
