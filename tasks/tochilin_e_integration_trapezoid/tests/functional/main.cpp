#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <tuple>

#include "tochilin_e_integration_trapezoid/common/include/common.hpp"
#include "tochilin_e_integration_trapezoid/mpi/include/ops_mpi.hpp"
#include "tochilin_e_integration_trapezoid/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tochilin_e_integration_trapezoid {

class TochilinEIntegrationTrapezoidFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_id = std::get<0>(params);

    input_data_.lower_bound = 0.0;
    input_data_.upper_bound = 1.0;
    input_data_.num_intervals = 1000;
    input_data_.function = [](double x) { return x * x; };

    expected_result_ = 1.0 / 3.0;

    if (test_id == 1) {
      input_data_.function = [](double x) { return x; };
      expected_result_ = 0.5;
    } else if (test_id == 2) {
      input_data_.function = [](double x) { return x * x; };
      expected_result_ = 1.0 / 3.0;
    } else if (test_id == 3) {
      input_data_.function = [](double x) { return x * x * x; };
      expected_result_ = 0.25;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double tolerance = 1e-4;
    return std::abs(output_data - expected_result_) < tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double expected_result_ = 0.0;
};

namespace {

TEST_P(TochilinEIntegrationTrapezoidFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "linear"), std::make_tuple(2, "quadratic"),
                                            std::make_tuple(3, "cubic")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TochilinEIntegrationTrapezoidMPI, InType>(
                                               kTestParam, PPC_SETTINGS_tochilin_e_integration_trapezoid),
                                           ppc::util::AddFuncTask<TochilinEIntegrationTrapezoidSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_tochilin_e_integration_trapezoid));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    TochilinEIntegrationTrapezoidFuncTests::PrintFuncTestName<TochilinEIntegrationTrapezoidFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TochilinEIntegrationTrapezoidFuncTests, kGtestValues, kPerfTestName);

}  // namespace

class TochilinEIntegrationTrapezoidValidationTests : public ::testing::Test {
 protected:
  static void RunValidationTest(const InType &input, bool expected_result) {
    TochilinEIntegrationTrapezoidSEQ task(input);
    EXPECT_EQ(task.Validation(), expected_result);
    if (!expected_result) {
      return;
    }
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
  }
};

TEST_F(TochilinEIntegrationTrapezoidValidationTests, InvalidZeroIntervals) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 1.0;
  input.num_intervals = 0;
  input.function = [](double x) { return x; };
  RunValidationTest(input, false);
}

TEST_F(TochilinEIntegrationTrapezoidValidationTests, InvalidNegativeIntervals) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 1.0;
  input.num_intervals = -5;
  input.function = [](double x) { return x; };
  RunValidationTest(input, false);
}

TEST_F(TochilinEIntegrationTrapezoidValidationTests, InvalidReversedBounds) {
  InType input;
  input.lower_bound = 5.0;
  input.upper_bound = 1.0;
  input.num_intervals = 100;
  input.function = [](double x) { return x; };
  RunValidationTest(input, false);
}

TEST_F(TochilinEIntegrationTrapezoidValidationTests, InvalidNullFunction) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 1.0;
  input.num_intervals = 100;
  input.function = nullptr;
  RunValidationTest(input, false);
}

TEST_F(TochilinEIntegrationTrapezoidValidationTests, ValidInput) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 1.0;
  input.num_intervals = 100;
  input.function = [](double x) { return x * x; };
  RunValidationTest(input, true);
}

class TochilinEIntegrationTrapezoidAccuracyTests : public ::testing::Test {
 protected:
  static void RunAccuracyTest(const InType &input, double expected, double tolerance) {
    TochilinEIntegrationTrapezoidSEQ seq_task(input);
    ASSERT_TRUE(seq_task.Validation());
    ASSERT_TRUE(seq_task.PreProcessing());
    ASSERT_TRUE(seq_task.Run());
    ASSERT_TRUE(seq_task.PostProcessing());
    double seq_result = seq_task.GetOutput();
    EXPECT_NEAR(seq_result, expected, tolerance);
  }
};

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateConstant) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 5.0;
  input.num_intervals = 100;
  input.function = [](double) { return 3.0; };
  RunAccuracyTest(input, 15.0, 1e-10);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateLinear) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 4.0;
  input.num_intervals = 1000;
  input.function = [](double x) { return (2.0 * x) + 1.0; };
  RunAccuracyTest(input, 20.0, 1e-6);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateSin) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = std::numbers::pi;
  input.num_intervals = 10000;
  input.function = [](double x) { return std::sin(x); };
  RunAccuracyTest(input, 2.0, 1e-6);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateCos) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = std::numbers::pi / 2.0;
  input.num_intervals = 10000;
  input.function = [](double x) { return std::cos(x); };
  RunAccuracyTest(input, 1.0, 1e-6);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateExp) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 1.0;
  input.num_intervals = 10000;
  input.function = [](double x) { return std::exp(x); };
  RunAccuracyTest(input, std::numbers::e - 1.0, 1e-6);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegratePolynomial) {
  InType input;
  input.lower_bound = -1.0;
  input.upper_bound = 1.0;
  input.num_intervals = 10000;
  input.function = [](double x) { return (x * x * x * x) - (2.0 * x * x) + 1.0; };
  RunAccuracyTest(input, 2.0 * ((1.0 / 5.0) - (2.0 / 3.0) + 1.0), 1e-5);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateNegativeBounds) {
  InType input;
  input.lower_bound = -2.0;
  input.upper_bound = 0.0;
  input.num_intervals = 1000;
  input.function = [](double x) { return x * x; };
  RunAccuracyTest(input, 8.0 / 3.0, 1e-5);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateSmallInterval) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 0.001;
  input.num_intervals = 100;
  input.function = [](double x) { return x; };
  RunAccuracyTest(input, 0.0000005, 1e-10);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateLargeInterval) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 100.0;
  input.num_intervals = 100000;
  input.function = [](double x) { return x; };
  RunAccuracyTest(input, 5000.0, 1e-3);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateSqrt) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 4.0;
  input.num_intervals = 10000;
  input.function = [](double x) { return std::sqrt(x); };
  RunAccuracyTest(input, 16.0 / 3.0, 1e-4);
}

TEST_F(TochilinEIntegrationTrapezoidAccuracyTests, IntegrateMinimalIntervals) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 1.0;
  input.num_intervals = 1;
  input.function = [](double x) { return x; };
  RunAccuracyTest(input, 0.5, 1e-10);
}

class TochilinEIntegrationTrapezoidMpiTests : public ::testing::Test {
 protected:
  static void RunMpiTest(const InType &input, double expected, double tolerance) {
    TochilinEIntegrationTrapezoidMPI mpi_task(input);
    ASSERT_TRUE(mpi_task.Validation());
    ASSERT_TRUE(mpi_task.PreProcessing());
    ASSERT_TRUE(mpi_task.Run());
    ASSERT_TRUE(mpi_task.PostProcessing());

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      double mpi_result = mpi_task.GetOutput();
      EXPECT_NEAR(mpi_result, expected, tolerance);
    }
  }
};

TEST_F(TochilinEIntegrationTrapezoidMpiTests, MpiIntegrateQuadratic) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 1.0;
  input.num_intervals = 10000;
  input.function = [](double x) { return x * x; };
  RunMpiTest(input, 1.0 / 3.0, 1e-6);
}

TEST_F(TochilinEIntegrationTrapezoidMpiTests, MpiIntegrateSin) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = std::numbers::pi;
  input.num_intervals = 10000;
  input.function = [](double x) { return std::sin(x); };
  RunMpiTest(input, 2.0, 1e-6);
}

TEST_F(TochilinEIntegrationTrapezoidMpiTests, MpiIntegrateExp) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 1.0;
  input.num_intervals = 10000;
  input.function = [](double x) { return std::exp(x); };
  RunMpiTest(input, std::numbers::e - 1.0, 1e-6);
}

TEST_F(TochilinEIntegrationTrapezoidMpiTests, MpiIntegrateConstant) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 10.0;
  input.num_intervals = 1000;
  input.function = [](double) { return 5.0; };
  RunMpiTest(input, 50.0, 1e-10);
}

TEST_F(TochilinEIntegrationTrapezoidMpiTests, MpiIntegrateCubic) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 2.0;
  input.num_intervals = 10000;
  input.function = [](double x) { return x * x * x; };
  RunMpiTest(input, 4.0, 1e-5);
}

class TochilinEIntegrationTrapezoidCompareTests : public ::testing::Test {
 protected:
  static void CompareSeqAndMpi(const InType &input, double tolerance) {
    TochilinEIntegrationTrapezoidSEQ seq_task(input);
    ASSERT_TRUE(seq_task.Validation());
    ASSERT_TRUE(seq_task.PreProcessing());
    ASSERT_TRUE(seq_task.Run());
    ASSERT_TRUE(seq_task.PostProcessing());
    double seq_result = seq_task.GetOutput();

    TochilinEIntegrationTrapezoidMPI mpi_task(input);
    ASSERT_TRUE(mpi_task.Validation());
    ASSERT_TRUE(mpi_task.PreProcessing());
    ASSERT_TRUE(mpi_task.Run());
    ASSERT_TRUE(mpi_task.PostProcessing());

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      double mpi_result = mpi_task.GetOutput();
      EXPECT_NEAR(seq_result, mpi_result, tolerance);
    }
  }
};

TEST_F(TochilinEIntegrationTrapezoidCompareTests, CompareQuadratic) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 1.0;
  input.num_intervals = 1000;
  input.function = [](double x) { return x * x; };
  CompareSeqAndMpi(input, 1e-10);
}

TEST_F(TochilinEIntegrationTrapezoidCompareTests, CompareSin) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = std::numbers::pi;
  input.num_intervals = 1000;
  input.function = [](double x) { return std::sin(x); };
  CompareSeqAndMpi(input, 1e-10);
}

TEST_F(TochilinEIntegrationTrapezoidCompareTests, CompareExp) {
  InType input;
  input.lower_bound = 0.0;
  input.upper_bound = 2.0;
  input.num_intervals = 1000;
  input.function = [](double x) { return std::exp(x); };
  CompareSeqAndMpi(input, 1e-10);
}

TEST_F(TochilinEIntegrationTrapezoidCompareTests, ComparePolynomial) {
  InType input;
  input.lower_bound = -1.0;
  input.upper_bound = 2.0;
  input.num_intervals = 1000;
  input.function = [](double x) { return ((3.0 * x * x) - (2.0 * x)) + 5.0; };
  CompareSeqAndMpi(input, 1e-10);
}

}  // namespace tochilin_e_integration_trapezoid
