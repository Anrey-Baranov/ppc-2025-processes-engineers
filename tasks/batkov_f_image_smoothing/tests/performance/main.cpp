#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "batkov_f_image_smoothing/common/include/common.hpp"
#include "batkov_f_image_smoothing/mpi/include/ops_mpi.hpp"
#include "batkov_f_image_smoothing/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace batkov_f_image_smoothing {

class BatkovFRunPerfTestProcesses2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  float preprocess_blur_value_ = 0.0F;

  void SetUp() override {
    size_t width = 7680;
    size_t height = 4320;
    size_t channels = 3;

    input_data_.width = width;
    input_data_.height = height;
    input_data_.channels = channels;

    input_data_.data.resize(width * height * channels);

    for (size_t col = 0; col < width; col += channels) {
      for (size_t row = 0; row < height; row += channels) {
        unsigned int hash_r = 2654435761U ^ (row * 73856093U);
        unsigned int hash_g = 2654435761U ^ (col * 19349663U);
        unsigned int hash_b = 2654435761U ^ ((row + col) * 83492791U);

        hash_r = (hash_r ^ (hash_r >> 13)) * 2654435761U;
        hash_g = (hash_g ^ (hash_g >> 13)) * 2654435761U;
        hash_b = (hash_b ^ (hash_b >> 13)) * 2654435761U;

        size_t index = (col * height) + row;
        input_data_.data[index + 0] = static_cast<uint8_t>((hash_r >> 8) & 0xFF);
        input_data_.data[index + 1] = static_cast<uint8_t>((hash_g >> 8) & 0xFF);
        input_data_.data[index + 2] = static_cast<uint8_t>((hash_b >> 8) & 0xFF);
      }
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
};

TEST_P(BatkovFRunPerfTestProcesses2, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BatkovFImageSmoothingMPI, BatkovFImageSmoothingSEQ>(
    PPC_SETTINGS_batkov_f_image_smoothing);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BatkovFRunPerfTestProcesses2::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BatkovFRunPerfTestProcesses2, kGtestValues, kPerfTestName);

}  // namespace batkov_f_image_smoothing
