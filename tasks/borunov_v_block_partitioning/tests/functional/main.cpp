#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>

#include "borunov_v_block_partitioning/common/include/common.hpp"
#include "borunov_v_block_partitioning/mpi/include/ops_mpi.hpp"
#include "borunov_v_block_partitioning/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace borunov_v_block_partitioning {

class BorunovLinearFilterTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<typename BorunovLinearFilterTest::ParamType> &info) {
    std::string task_name = std::get<1>(info.param);
    auto test_params = std::get<2>(info.param);

    int width = std::get<0>(test_params);
    int height = std::get<1>(test_params);

    std::string name = task_name + "_" + std::to_string(width) + "x" + std::to_string(height);

    std::ranges::replace(name, ':', '_');
    std::ranges::replace(name, '.', '_');
    std::ranges::replace(name, '/', '_');

    return name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<2>(GetParam());
    int width = std::get<0>(params);
    int height = std::get<1>(params);

    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    input_data_.resize(static_cast<std::size_t>(2) + pixels);
    input_data_[0] = width;
    input_data_[1] = height;

    const int pattern = (width + height) % 4;
    int *pixels_ptr = input_data_.data() + 2;

    if (pattern == 0) {
      std::mt19937 gen(static_cast<unsigned int>(width * height) + 12345U);
      std::uniform_int_distribution<int> dist(0, 255);
      for (std::size_t i = 0; i < pixels; ++i) {
        pixels_ptr[i] = dist(gen);
      }
    } else if (pattern == 1) {
      std::fill(pixels_ptr, pixels_ptr + pixels, 0);
      const int cy = height / 2;
      const int cx = width / 2;
      pixels_ptr[(cy * width) + cx] = 255;
    } else if (pattern == 2) {
      for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
          pixels_ptr[(row * width) + col] = (((col + row) & 1) != 0) ? 0 : 255;
        }
      }
    } else {
      for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
          pixels_ptr[(row * width) + col] = (col * 255) / std::max(1, width - 1);
        }
      }
    }

    CalculateReferenceOutput(width, height);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == reference_output_;
  }

 private:
  InType input_data_;
  OutType reference_output_;

  void CalculateReferenceOutput(int width, int height) {
    reference_output_.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    const std::array<std::array<float, 3>, 3> kernel = {{
        {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
        {2.0F / 16.0F, 4.0F / 16.0F, 2.0F / 16.0F},
        {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
    }};

    const int *pixels = input_data_.data() + 2;

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        const int x0 = std::clamp(j - 1, 0, width - 1);
        const int x1 = j;
        const int x2 = std::clamp(j + 1, 0, width - 1);

        const int y0 = std::clamp(i - 1, 0, height - 1);
        const int y1 = i;
        const int y2 = std::clamp(i + 1, 0, height - 1);

        float sum = 0.0F;

        sum += static_cast<float>(pixels[(y0 * width) + x0]) * kernel[0][0];
        sum += static_cast<float>(pixels[(y0 * width) + x1]) * kernel[0][1];
        sum += static_cast<float>(pixels[(y0 * width) + x2]) * kernel[0][2];

        sum += static_cast<float>(pixels[(y1 * width) + x0]) * kernel[1][0];
        sum += static_cast<float>(pixels[(y1 * width) + x1]) * kernel[1][1];
        sum += static_cast<float>(pixels[(y1 * width) + x2]) * kernel[1][2];

        sum += static_cast<float>(pixels[(y2 * width) + x0]) * kernel[2][0];
        sum += static_cast<float>(pixels[(y2 * width) + x1]) * kernel[2][1];
        sum += static_cast<float>(pixels[(y2 * width) + x2]) * kernel[2][2];

        reference_output_[(i * width) + j] = static_cast<int>(std::round(sum));
      }
    }
  }
};

TEST_P(BorunovLinearFilterTest, RunFilter) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParams = {
    std::make_tuple(1, 1),     std::make_tuple(1, 64),   std::make_tuple(64, 1),    std::make_tuple(3, 5),
    std::make_tuple(5, 3),     std::make_tuple(10, 10),  std::make_tuple(20, 15),   std::make_tuple(15, 20),
    std::make_tuple(32, 32),   std::make_tuple(31, 29),  std::make_tuple(128, 128), std::make_tuple(256, 128),
    std::make_tuple(128, 256), std::make_tuple(256, 256)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BorunovVBlockPartitioningMPI, InType>(
                                               kTestParams, PPC_SETTINGS_borunov_v_block_partitioning),
                                           ppc::util::AddFuncTask<BorunovVBlockPartitioningSEQ, InType>(
                                               kTestParams, PPC_SETTINGS_borunov_v_block_partitioning));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(BorunovFilterTests, BorunovLinearFilterTest, kGtestValues,
                         BorunovLinearFilterTest::PrintTestParam);

}  // namespace borunov_v_block_partitioning
