#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "batkov_f_image_smoothing/common/include/common.hpp"
#include "batkov_f_image_smoothing/mpi/include/ops_mpi.hpp"
#include "batkov_f_image_smoothing/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace batkov_f_image_smoothing {

class BatkovFRunFuncTestsProcesses2 : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string p0 = std::get<0>(test_param);
    std::string p1 = std::to_string(std::get<1>(test_param));
    std::string p2 = std::to_string(std::get<2>(test_param));
    std::string p3 = std::to_string(std::get<3>(test_param));
    return p0 + "_" + p1 + "x" + p2 + "x" + p3;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_.width = std::get<1>(params);
    input_data_.height = std::get<2>(params);
    input_data_.channels = std::get<3>(params);

    size_t size = input_data_.width * input_data_.height * input_data_.channels;
    input_data_.data.resize(size);

    for (size_t i = 0; i < size; ++i) {
      input_data_.data[i] = dis_(gen_);
    }

    preprocess_blur_value_ = CalcLaplacianVariance(input_data_);
  }

  static float CalcLaplacianVariance(const Image &image) {
    std::vector<float> gray(image.width * image.height);

    const auto &data = image.data;
    size_t width = image.width;
    size_t height = image.height;
    size_t channels = image.channels;

    if (channels == 1) {
      for (size_t i = 0; i < width * height; i++) {
        gray[i] = static_cast<float>(data[i]);
      }
    } else {
      for (size_t i = 0; i < width * height; i++) {
        size_t idx = i * channels;
        auto r = static_cast<float>(data[idx + 0]);
        auto g = static_cast<float>(data[idx + 1]);
        auto b = static_cast<float>(data[idx + 2]);

        gray[i] = (0.299F * r) + (0.587F * g) + (0.114F * b);
      }
    }

    std::vector<float> laplacian(width * height, 0.0F);
    for (size_t y_px = 1; y_px < height - 1; y_px++) {
      for (size_t x_px = 1; x_px < width - 1; x_px++) {
        size_t idx = (y_px * width) + x_px;

        float value = -gray[((y_px - 1) * width) + x_px] - gray[(y_px * width) + (x_px - 1)] + (4.0F * gray[idx]) -
                      gray[(y_px * width) + (x_px + 1)] - gray[((y_px + 1) * width) + x_px];

        laplacian[idx] = value;
      }
    }

    float mean = 0.0F;
    for (size_t i = 0; i < width * height; i++) {
      mean += laplacian[i];
    }
    mean /= static_cast<float>(width * height);

    float variance = 0.0F;
    for (size_t i = 0; i < width * height; i++) {
      float diff = laplacian[i] - mean;
      variance += diff * diff;
    }
    variance /= static_cast<float>(width * height);

    return variance;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    float post_process_blur_value = CalcLaplacianVariance(output_data);

    return (preprocess_blur_value_ / post_process_blur_value) > 2.0F;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_{rd_()};
  std::uniform_int_distribution<size_t> dis_{0, 255};

  float preprocess_blur_value_ = 0.0F;
  InType input_data_;
};

namespace {

TEST_P(BatkovFRunFuncTestsProcesses2, ImageSmoothing) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    std::make_tuple("tiny_image", 10, 10, 3), std::make_tuple("small_image", 50, 50, 3),
    std::make_tuple("medium_image", 150, 150, 3), std::make_tuple("big_image", 300, 300, 3)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BatkovFImageSmoothingSEQ, InType>(kTestParam, PPC_SETTINGS_batkov_f_image_smoothing),
    ppc::util::AddFuncTask<BatkovFImageSmoothingMPI, InType>(kTestParam, PPC_SETTINGS_batkov_f_image_smoothing));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BatkovFRunFuncTestsProcesses2::PrintFuncTestName<BatkovFRunFuncTestsProcesses2>;

INSTANTIATE_TEST_SUITE_P(ImageSmoothingTests, BatkovFRunFuncTestsProcesses2, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace batkov_f_image_smoothing
