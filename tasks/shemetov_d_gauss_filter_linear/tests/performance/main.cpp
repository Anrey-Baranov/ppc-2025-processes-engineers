#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"
#include "shemetov_d_gauss_filter_linear/mpi/include/ops_mpi.hpp"
#include "shemetov_d_gauss_filter_linear/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shemetov_d_gauss_filter_linear {

class ShemetovDGaussFilterPerformanceTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data;

  void SetUp() override {
    const int size = 1024;

    Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

    input_data.assign(size, std::vector<Pixel>(size, m_pixel));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }
    if (output_data.size() != input_data.size()) {
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
};

TEST_F(ShemetovDGaussFilterPerformanceTests, SeqFullCycle) {
  GaussFilterSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST_F(ShemetovDGaussFilterPerformanceTests, MpiFullCycle) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GaussFilterMPI task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

TEST_F(ShemetovDGaussFilterPerformanceTests, SeqRunOnly) {
  GaussFilterSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
}

TEST_F(ShemetovDGaussFilterPerformanceTests, MpiRunOnly) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GaussFilterMPI task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());

  SUCCEED();
}

TEST(ShemetovDGaussFilterPerformanceExtraTests, SeqSmallData) {
  Pixel m_pixel = {.channel_red = 50, .channel_green = 50, .channel_blue = 50};

  InType data(10, std::vector<Pixel>(10, m_pixel));

  GaussFilterSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(ShemetovDGaussFilterPerformanceExtraTests, MpiSmallData) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Pixel m_pixel = {.channel_red = 50, .channel_green = 50, .channel_blue = 50};

  InType data(10, std::vector<Pixel>(10, m_pixel));

  GaussFilterMPI task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

TEST(ShemetovDGaussFilterPerformanceExtraTests, SeqLargeData) {
  const int size = 1024;

  Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

  InType data(size, std::vector<Pixel>(size, m_pixel));

  GaussFilterSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(ShemetovDGaussFilterPerformanceExtraTests, MpiLargeData) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int size = 1024;

  Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

  InType data(size, std::vector<Pixel>(size, m_pixel));

  GaussFilterMPI task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

TEST(ShemetovDGaussFilterPerformanceExtraTests, SeqVariousSizes) {
  std::vector<int> sizes = {16, 64, 128, 512, 1024};
  for (int size : sizes) {
    Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

    InType data(size, std::vector<Pixel>(size, m_pixel));

    GaussFilterSEQ task(data);
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
  }
}

TEST(ShemetovDGaussFilterPerformanceExtraTests, MpiVariousSizes) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> sizes = {16, 64, 128, 512, 1024};
  for (int size : sizes) {
    Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

    InType data(size, std::vector<Pixel>(size, m_pixel));

    GaussFilterMPI task(data);
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
  }
}

TEST(ShemetovDGaussFilterPerformanceExtraTests, SeqLargeRGB) {
  const int size = 2048;

  Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

  InType data(size, std::vector<Pixel>(size, m_pixel));

  GaussFilterSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(ShemetovDGaussFilterPerformanceExtraTests, MpiLargeRGB) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int size = 2048;

  Pixel m_pixel = {.channel_red = 128, .channel_green = 128, .channel_blue = 128};

  InType data(size, std::vector<Pixel>(size, m_pixel));

  GaussFilterMPI task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

}  // namespace shemetov_d_gauss_filter_linear
