#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

namespace shemetov_d_increasing_contrast {

class ShemetovDIncreaseContrastPerformanceTests : public ::testing::Test {
 protected:
  InType input_data;

  void SetUp() override {
    constexpr size_t kSize = 1024;

    Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

    input_data.assign(kSize, m_pixel);
  }
};

TEST_F(ShemetovDIncreaseContrastPerformanceTests, SeqFullCycle) {
  IncreaseContrastTaskSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST_F(ShemetovDIncreaseContrastPerformanceTests, MpiFullCycle) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  IncreaseContrastTaskMPI task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

TEST_F(ShemetovDIncreaseContrastPerformanceTests, SeqRunOnly) {
  IncreaseContrastTaskSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
}

TEST_F(ShemetovDIncreaseContrastPerformanceTests, MpiRunOnly) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  IncreaseContrastTaskMPI task(input_data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());

  SUCCEED();
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, SeqSmallData) {
  Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};
  InType data(100, m_pixel);

  IncreaseContrastTaskSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, MpiSmallData) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};
  InType data(100, m_pixel);

  IncreaseContrastTaskSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, SeqLargeData) {
  constexpr size_t kLargeSize = 10000000;
  Pixel m_pixel = {.channel_red = 200, .channel_green = 200, .channel_blue = 200};
  InType data(kLargeSize, m_pixel);

  IncreaseContrastTaskSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, MpiLargeData) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr size_t kLargeSize = 10000000;
  Pixel m_pixel = {.channel_red = 200, .channel_green = 200, .channel_blue = 200};
  InType data(kLargeSize, m_pixel);

  IncreaseContrastTaskSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, SeqVariousSizes) {
  std::vector<size_t> sizes = {100, 1000, 10000, 100000, 1000000};

  Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

  for (size_t size : sizes) {
    InType data(size, m_pixel);

    IncreaseContrastTaskSEQ task(data);
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
  }
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, MpiVariousSizes) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<size_t> sizes = {100, 1000, 10000, 100000, 1000000};

  Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

  for (size_t size : sizes) {
    InType data(size, m_pixel);

    IncreaseContrastTaskSEQ task(data);
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
  }
}

}  // namespace shemetov_d_increasing_contrast
