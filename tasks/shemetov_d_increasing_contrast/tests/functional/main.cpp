#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shemetov_d_increasing_contrast {

class ShemetovDIncreaseContrastFunctionalTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  InType input_data;
  int width = 0;
  int height = 0;
  int channels = 0;

  void SetUp() override {
    const auto &test_param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string image_path = std::get<1>(test_param);

    uint8_t *data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    ASSERT_TRUE(data != nullptr) << "Failed to load image: " << image_path;

    const size_t total_size = static_cast<size_t>(width) * static_cast<size_t>(height);
    input_data.resize(total_size);

    for (size_t i = 0; i < total_size; ++i) {
      const size_t global_idx = i * 3;
      input_data[i] = Pixel{
          .channel_red = data[global_idx], .channel_green = data[global_idx + 1], .channel_blue = data[global_idx + 2]};
    }

    stbi_image_free(data);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    if (output_data.size() != input_data.size()) {
      return false;
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

TEST(ShemetovDIncreaseContrastFunctionalExtraTests, SmallSyntheticSEQ) {
  Pixel m_pixel = {.channel_red = 10, .channel_green = 20, .channel_blue = 30};
  InType input(25, m_pixel);

  IncreaseContrastTaskSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(ShemetovDIncreaseContrastFunctionalExtraTests, SmallSyntheticMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Pixel m_pixel = {.channel_red = 10, .channel_green = 20, .channel_blue = 30};
  InType input(25, m_pixel);

  IncreaseContrastTaskMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

TEST(ShemetovDIncreaseContrastFunctionalExtraTests, SinglePixelMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Pixel m_pixel = {.channel_red = 100, .channel_green = 100, .channel_blue = 100};
  InType input = {m_pixel};

  IncreaseContrastTaskMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    const OutType &out = task.GetOutput();

    EXPECT_EQ(out[0].channel_red, 130);
    EXPECT_EQ(out[0].channel_green, 130);
    EXPECT_EQ(out[0].channel_blue, 130);
  }
}

TEST(ShemetovDIncreaseContrastFunctionalExtraTests, GradientImageSEQ) {
  InType input(256);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = {.channel_red = static_cast<uint8_t>(i),
                .channel_green = static_cast<uint8_t>(i),
                .channel_blue = static_cast<uint8_t>(i)};
  }

  IncreaseContrastTaskSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST_P(ShemetovDIncreaseContrastFunctionalTests, FullCycle) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    std::make_tuple("Image_0", "tasks/shemetov_d_increasing_contrast/data/pic_0.jpg"),
    std::make_tuple("Image_1", "tasks/shemetov_d_increasing_contrast/data/pic_1.jpg"),
    std::make_tuple("Image_2", "tasks/shemetov_d_increasing_contrast/data/pic_2.jpg"),
    std::make_tuple("Image_3", "tasks/shemetov_d_increasing_contrast/data/pic_3.jpg")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IncreaseContrastTaskMPI, InType>(kTestParam, PPC_SETTINGS_shemetov_d_increasing_contrast),
    ppc::util::AddFuncTask<IncreaseContrastTaskSEQ, InType>(kTestParam, PPC_SETTINGS_shemetov_d_increasing_contrast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName =
    ShemetovDIncreaseContrastFunctionalTests::PrintFuncTestName<ShemetovDIncreaseContrastFunctionalTests>;

INSTANTIATE_TEST_SUITE_P(ImageTests, ShemetovDIncreaseContrastFunctionalTests, kGtestValues, kTestName);

}  // namespace shemetov_d_increasing_contrast
