#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "marin_l_linear_filter_vertical/common/include/common.hpp"
#include "marin_l_linear_filter_vertical/mpi/include/ops_mpi.hpp"
#include "marin_l_linear_filter_vertical/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace marin_l_linear_filter_vertical {

namespace {

constexpr std::array<std::array<int, 3>, 3> kGaussKernel = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
constexpr int kKernelSum = 16;

std::vector<uint8_t> ApplyGaussianFilter(const std::vector<uint8_t> &input, int width, int height) {
  std::vector<uint8_t> output(input.size());

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int sum = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int ny = row + ky;
          int nx = col + kx;
          uint8_t pixel_value = 0;
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            pixel_value = input[(ny * width) + nx];
          }
          sum += pixel_value * kGaussKernel.at(ky + 1).at(kx + 1);
        }
      }
      output[(row * width) + col] = static_cast<uint8_t>(std::clamp(sum / kKernelSum, 0, 255));
    }
  }
  return output;
}

ImageData CreateUniformImage(int width, int height, uint8_t value) {
  ImageData img;
  img.width = width;
  img.height = height;
  img.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height), value);
  return img;
}

ImageData CreateGradientImage(int width, int height) {
  ImageData img;
  img.width = width;
  img.height = height;
  img.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      img.pixels[(row * width) + col] = static_cast<uint8_t>((col + row) % 256);
    }
  }
  return img;
}

ImageData CreateCheckerboardImage(int width, int height) {
  ImageData img;
  img.width = width;
  img.height = height;
  img.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      img.pixels[(row * width) + col] = (((col + row) % 2) == 0) ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
    }
  }
  return img;
}

ImageData CreateVerticalStripesImage(int width, int height) {
  ImageData img;
  img.width = width;
  img.height = height;
  img.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      img.pixels[(row * width) + col] = ((col % 2) == 0) ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
    }
  }
  return img;
}

ImageData CreateHorizontalStripesImage(int width, int height) {
  ImageData img;
  img.width = width;
  img.height = height;
  img.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      img.pixels[(row * width) + col] = ((row % 2) == 0) ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
    }
  }
  return img;
}

ImageData CreateRandomImage(int width, int height, unsigned int seed) {
  ImageData img;
  img.width = width;
  img.height = height;
  img.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &pixel : img.pixels) {
    pixel = static_cast<uint8_t>(dist(gen));
  }
  return img;
}

}  // namespace

class MarinLLinearFilterVerticalFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_id = std::get<0>(params);

    switch (test_id) {
      case 1:
        input_data_ = CreateUniformImage(10, 10, 128);
        break;
      case 2:
        input_data_ = CreateGradientImage(15, 15);
        break;
      case 3:
        input_data_ = CreateCheckerboardImage(12, 12);
        break;
      case 4:
        input_data_ = CreateVerticalStripesImage(20, 10);
        break;
      case 5:
        input_data_ = CreateHorizontalStripesImage(10, 20);
        break;
      case 6:
        input_data_ = CreateRandomImage(25, 25, 42);
        break;
      case 7:
        input_data_ = CreateUniformImage(3, 3, 200);
        break;
      case 8:
        input_data_ = CreateUniformImage(1, 1, 100);
        break;
      case 9:
        input_data_ = CreateGradientImage(100, 100);
        break;
      case 10:
        input_data_ = CreateUniformImage(5, 1, 50);
        break;
      case 11:
        input_data_ = CreateUniformImage(1, 5, 75);
        break;
      case 12:
        input_data_ = CreateRandomImage(50, 50, 123);
        break;
      default:
        input_data_ = CreateUniformImage(10, 10, 128);
        break;
    }

    expected_output_ = ApplyGaussianFilter(input_data_.pixels, input_data_.width, input_data_.height);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<uint8_t> expected_output_;
};

namespace {

TEST_P(MarinLLinearFilterVerticalFuncTests, GaussianFilterTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {std::make_tuple(1, "uniform_10x10"),
                                             std::make_tuple(2, "gradient_15x15"),
                                             std::make_tuple(3, "checkerboard_12x12"),
                                             std::make_tuple(4, "vertical_stripes_20x10"),
                                             std::make_tuple(5, "horizontal_stripes_10x20"),
                                             std::make_tuple(6, "random_25x25"),
                                             std::make_tuple(7, "small_3x3"),
                                             std::make_tuple(8, "minimal_1x1"),
                                             std::make_tuple(9, "large_100x100"),
                                             std::make_tuple(10, "row_5x1"),
                                             std::make_tuple(11, "column_1x5"),
                                             std::make_tuple(12, "random_50x50")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<MarinLLinearFilterVerticalMPI, InType>(
                                               kTestParam, PPC_SETTINGS_marin_l_linear_filter_vertical),
                                           ppc::util::AddFuncTask<MarinLLinearFilterVerticalSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_marin_l_linear_filter_vertical));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = MarinLLinearFilterVerticalFuncTests::PrintFuncTestName<MarinLLinearFilterVerticalFuncTests>;

INSTANTIATE_TEST_SUITE_P(GaussianFilterTests, MarinLLinearFilterVerticalFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace marin_l_linear_filter_vertical
