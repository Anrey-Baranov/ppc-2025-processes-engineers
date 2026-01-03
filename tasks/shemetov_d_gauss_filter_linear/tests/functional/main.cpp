#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"
#include "shemetov_d_gauss_filter_linear/mpi/include/ops_mpi.hpp"
#include "shemetov_d_gauss_filter_linear/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shemetov_d_gauss_filter_linear {

class ShemetovDGaussFilterFunctionalTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  InType input_data;

  int width = 0;
  int height = 0;
  int channels = 0;
  int image_size = 0;

  void SetUp() override {
    const auto &test_param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string image_path = std::get<1>(test_param);

    uint8_t *data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    ASSERT_TRUE(static_cast<bool>(data != nullptr)) << "Failed to load image: " << image_path;

    std::vector<uint8_t> image(data, data + static_cast<ptrdiff_t>(width * height * 3));
    stbi_image_free(data);

    input_data.clear();
    input_data.reserve(height);
    for (int i = 0; i < height; ++i) {
      input_data.emplace_back(width);
    }

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        const auto input_idx = static_cast<size_t>((i * width) + j) * 3;

        input_data[i][j] = {.channel_red = image[input_idx],
                            .channel_green = image[input_idx + 1],
                            .channel_blue = image[input_idx + 2]};
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty() || output_data.size() != input_data.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i].size() != input_data[i].size()) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }

 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }
};

TEST(ShemetovDGaussFilterFunctionalExtraTests, SmallSyntheticSEQ) {
  Pixel m_pixel = {.channel_red = 10, .channel_green = 10, .channel_blue = 10};

  InType input(5, std::vector<Pixel>(5, m_pixel));

  input[2][2] = {.channel_red = 200, .channel_green = 200, .channel_blue = 200};

  GaussFilterSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(ShemetovDGaussFilterFunctionalExtraTests, SmallSyntheticMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Pixel m_pixel = {.channel_red = 10, .channel_green = 10, .channel_blue = 10};

  InType input(5, std::vector<Pixel>(5, m_pixel));

  input[2][2] = {.channel_red = 200, .channel_green = 200, .channel_blue = 200};

  GaussFilterMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

TEST(ShemetovDGaussFilterFunctionalExtraTests, SmallestRGBImageMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Pixel m_pixel = {.channel_red = 100, .channel_green = 150, .channel_blue = 200};

  InType input(3, std::vector<Pixel>(3, m_pixel));

  GaussFilterMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_GE(task.GetOutput()[i][j].channel_red, 0);
        EXPECT_LE(task.GetOutput()[i][j].channel_red, 255);
      }
    }
  }
}

TEST(ShemetovDGaussFilterFunctionalExtraTests, SinglePixelMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Pixel m_pixel = {.channel_red = 50, .channel_green = 75, .channel_blue = 125};

  InType input(1, std::vector<Pixel>(1, m_pixel));

  GaussFilterMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    auto &out = task.GetOutput();
    EXPECT_EQ(out[0][0].channel_red, 50);
    EXPECT_EQ(out[0][0].channel_green, 75);
    EXPECT_EQ(out[0][0].channel_blue, 125);
  }
}

TEST(ShemetovDGaussFilterFunctionalExtraTests, GradientImageMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int height = 5;
  int width = 5;

  InType input(height, std::vector<Pixel>(width));
  for (int i = 0; i < height; ++i) {
    const auto value = static_cast<uint8_t>(i * 255 / (height - 1));

    for (int j = 0; j < width; ++j) {
      input[i][j] = {.channel_red = value, .channel_green = value, .channel_blue = value};
    }
  }

  GaussFilterMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    auto &out = task.GetOutput();
    EXPECT_GE(out[2][2].channel_red, 0);
    EXPECT_LE(out[2][2].channel_red, 255);
  }
}

TEST(ShemetovDGaussFilterFunctionalExtraTests, HorizontalWhiteLine) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Pixel m_pixel = {.channel_red = 0, .channel_green = 0, .channel_blue = 0};

  InType input(5, std::vector<Pixel>(5, m_pixel));

  for (int j = 0; j < 5; ++j) {
    input[2][j] = {.channel_red = 255, .channel_green = 255, .channel_blue = 255};
  }

  GaussFilterMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(ShemetovDGaussFilterFunctionalExtraTests, VerticalRedLine) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Pixel m_pixel = {.channel_red = 0, .channel_green = 0, .channel_blue = 0};

  InType input(5, std::vector<Pixel>(5, m_pixel));

  for (int i = 0; i < 5; ++i) {
    input[i][2] = {.channel_red = 255, .channel_green = 0, .channel_blue = 0};
  }

  GaussFilterMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST_P(ShemetovDGaussFilterFunctionalTests, FullCycle) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    std::make_tuple("Image_0", "tasks/shemetov_d_gauss_filter_linear/data/pic_0.jpg"),
    std::make_tuple("Image_1", "tasks/shemetov_d_gauss_filter_linear/data/pic_1.jpg"),
    std::make_tuple("Image_2", "tasks/shemetov_d_gauss_filter_linear/data/pic_2.jpg"),
    std::make_tuple("Image_3", "tasks/shemetov_d_gauss_filter_linear/data/pic_3.jpg")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_shemetov_d_gauss_filter_linear),
    ppc::util::AddFuncTask<GaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_shemetov_d_gauss_filter_linear));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = ShemetovDGaussFilterFunctionalTests::PrintFuncTestName<ShemetovDGaussFilterFunctionalTests>;

INSTANTIATE_TEST_SUITE_P(ImageTests, ShemetovDGaussFilterFunctionalTests, kGtestValues, kTestName);

}  // namespace shemetov_d_gauss_filter_linear
