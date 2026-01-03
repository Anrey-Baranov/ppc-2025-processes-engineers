#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "samoylenko_i_simple_iter_method/common/include/common.hpp"
#include "samoylenko_i_simple_iter_method/mpi/include/ops_mpi.hpp"
#include "samoylenko_i_simple_iter_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace samoylenko_i_simple_iter_method {

class SamoylenkoISimpleIterMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) override {
    int rank = 0;
    int is_mpi_init = 0;
    MPI_Initialized(&is_mpi_init);
    if (is_mpi_init != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    // only rank 0 has the answer
    if (rank > 0) {
      return true;
    }

    // solution vector should have size n and contain non-trivial values
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      return false;
    }

    int n = input_data_;
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
      double ax_i = 4.0 * output_data[i];
      if (i > 0) {
        ax_i += output_data[i - 1];
      }
      if (i < n - 1) {
        ax_i += output_data[i + 1];
      }
      max_diff = std::max(max_diff, std::fabs(ax_i - 1.0));
    }
    return max_diff < 1e-6;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(SamoylenkoISimpleIterMethodFuncTests, SimpleIterationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(1, "size_1"),   std::make_tuple(2, "size_2"),
                                             std::make_tuple(3, "size_3"),   std::make_tuple(5, "size_5"),
                                             std::make_tuple(7, "size_7"),   std::make_tuple(10, "size_10"),
                                             std::make_tuple(15, "size_15"), std::make_tuple(20, "size_20"),
                                             std::make_tuple(30, "size_30"), std::make_tuple(50, "size_50")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SamoylenkoISimpleIterMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_samoylenko_i_simple_iter_method),
                                           ppc::util::AddFuncTask<SamoylenkoISimpleIterMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_samoylenko_i_simple_iter_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    SamoylenkoISimpleIterMethodFuncTests::PrintFuncTestName<SamoylenkoISimpleIterMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(BasicTests, SamoylenkoISimpleIterMethodFuncTests, kGtestValues, kFuncTestName);

// Edge case tests for SEQ

TEST(SamoylenkoISimpleIterMethodEdgeCases, InvalidInputZeroSEQ) {
  SamoylenkoISimpleIterMethodSEQ task(0);
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, InvalidInputNegativeSEQ) {
  SamoylenkoISimpleIterMethodSEQ task(-5);
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, ValidInputPositiveSEQ) {
  SamoylenkoISimpleIterMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, PreProcessingSEQ) {
  SamoylenkoISimpleIterMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, FullExecutionSEQ) {
  SamoylenkoISimpleIterMethodSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput().size(), 5U);
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, MinimalSizeSEQ) {
  SamoylenkoISimpleIterMethodSEQ task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput().size(), 1U);
}

// Edge case tests for MPI

TEST(SamoylenkoISimpleIterMethodEdgeCases, InvalidInputZeroMPI) {
  SamoylenkoISimpleIterMethodMPI task(0);
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, InvalidInputNegativeMPI) {
  SamoylenkoISimpleIterMethodMPI task(-5);
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, ValidInputPositiveMPI) {
  SamoylenkoISimpleIterMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, PreProcessingMPI) {
  SamoylenkoISimpleIterMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, FullExecutionMPI) {
  SamoylenkoISimpleIterMethodMPI task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    EXPECT_EQ(task.GetOutput().size(), 5U);
  }
}

TEST(SamoylenkoISimpleIterMethodEdgeCases, MinimalSizeMPI) {
  SamoylenkoISimpleIterMethodMPI task(1);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    EXPECT_EQ(task.GetOutput().size(), 1U);
  }
}

}  // namespace

}  // namespace samoylenko_i_simple_iter_method
