#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "samoylenko_i_conj_grad_method/common/include/common.hpp"
#include "samoylenko_i_conj_grad_method/mpi/include/ops_mpi.hpp"
#include "samoylenko_i_conj_grad_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace samoylenko_i_conj_grad_method {

class SamoylenkoIConjGradMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param).first) + "_" + std::to_string(std::get<0>(test_param).second) + "_" +
           std::get<1>(test_param);
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
    if (output_data.size() != static_cast<size_t>(input_data_.first)) {
      return false;
    }

    size_t n = input_data_.first;
    int variant = input_data_.second;
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double ax_i = 0.0;
      switch (variant) {
        case 0: {
          ax_i = 4.0 * output_data[i];
          if (i > 0) {
            ax_i += output_data[i - 1];
          }
          if (i < n - 1) {
            ax_i += output_data[i + 1];
          }
          break;
        }

        case 1: {
          ax_i = 5.0 * output_data[i];
          break;
        }

        case 2: {
          ax_i = 3.0 * output_data[i];
          size_t j = n - 1 - i;
          if (i != j) {
            ax_i -= output_data[j];
          }
          break;
        }

        default:
          break;
      }

      max_diff = std::max(max_diff, std::fabs(ax_i - 1.0));
    }
    return max_diff < 1e-6;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_ = {0, 0};
};

namespace {

TEST_P(SamoylenkoIConjGradMethodFuncTests, ConjugateGradientTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {std::make_tuple(std::make_pair(1, 0), "size_1_variant_0"),
                                             std::make_tuple(std::make_pair(3, 0), "size_3_variant_0"),
                                             std::make_tuple(std::make_pair(10, 0), "size_10_variant_0"),
                                             std::make_tuple(std::make_pair(50, 0), "size_50_variant_0"),
                                             std::make_tuple(std::make_pair(1, 1), "size_1_variant_1"),
                                             std::make_tuple(std::make_pair(3, 1), "size_3_variant_1"),
                                             std::make_tuple(std::make_pair(10, 1), "size_10_variant_1"),
                                             std::make_tuple(std::make_pair(50, 1), "size_50_variant_1"),
                                             std::make_tuple(std::make_pair(1, 2), "size_1_variant_2"),
                                             std::make_tuple(std::make_pair(3, 2), "size_3_variant_2"),
                                             std::make_tuple(std::make_pair(10, 2), "size_10_variant_2"),
                                             std::make_tuple(std::make_pair(50, 2), "size_50_variant_2")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SamoylenkoIConjGradMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_samoylenko_i_conj_grad_method),
                                           ppc::util::AddFuncTask<SamoylenkoIConjGradMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_samoylenko_i_conj_grad_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = SamoylenkoIConjGradMethodFuncTests::PrintFuncTestName<SamoylenkoIConjGradMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(BasicTests, SamoylenkoIConjGradMethodFuncTests, kGtestValues, kFuncTestName);

// Edge case tests for SEQ

TEST(SamoylenkoIConjGradMethodEdgeCases, InvalidInputZeroSEQ) {
  SamoylenkoIConjGradMethodSEQ task({0, 0});
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, InvalidInputNegativeSEQ) {
  SamoylenkoIConjGradMethodSEQ task({-5, 0});
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, ValidInputPositiveSEQ) {
  SamoylenkoIConjGradMethodSEQ task({5, 0});
  EXPECT_TRUE(task.Validation());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, InvalidVariantSEQ) {
  SamoylenkoIConjGradMethodSEQ task({5, -1});
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, PreProcessingSEQ) {
  SamoylenkoIConjGradMethodSEQ task({5, 0});
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, FullExecutionSEQ) {
  SamoylenkoIConjGradMethodSEQ task({5, 0});
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput().size(), 5U);
}

TEST(SamoylenkoIConjGradMethodEdgeCases, MinimalSizeSEQ) {
  SamoylenkoIConjGradMethodSEQ task({1, 0});
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput().size(), 1U);
}

// Edge case tests for MPI

TEST(SamoylenkoIConjGradMethodEdgeCases, InvalidInputZeroMPI) {
  SamoylenkoIConjGradMethodMPI task({0, 0});
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, InvalidInputNegativeMPI) {
  SamoylenkoIConjGradMethodMPI task({-5, 0});
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, ValidInputPositiveMPI) {
  SamoylenkoIConjGradMethodMPI task({5, 0});
  EXPECT_TRUE(task.Validation());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, InvalidVariantMPI) {
  SamoylenkoIConjGradMethodMPI task({5, -1});
  EXPECT_FALSE(task.Validation());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, PreProcessingMPI) {
  SamoylenkoIConjGradMethodMPI task({5, 0});
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(SamoylenkoIConjGradMethodEdgeCases, FullExecutionMPI) {
  SamoylenkoIConjGradMethodMPI task({5, 0});
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

TEST(SamoylenkoIConjGradMethodEdgeCases, MinimalSizeMPI) {
  SamoylenkoIConjGradMethodMPI task({1, 0});
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

}  // namespace samoylenko_i_conj_grad_method
