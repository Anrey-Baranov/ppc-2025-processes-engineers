#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <string>
#include <tuple>
#include <vector>

#include "tochilin_e_integral_trapezium/common/include/common.hpp"
#include "tochilin_e_integral_trapezium/mpi/include/ops_mpi.hpp"
#include "tochilin_e_integral_trapezium/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tochilin_e_integral_trapezium {

class TochilinEIntegralTrapeziumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
        input_data_.lower_bounds = {0.0};
        input_data_.upper_bounds = {1.0};
        input_data_.num_steps = 100;
        input_data_.func = [](const std::vector<double> &) { return 1.0; };
        expected_result_ = 1.0;
        break;
      case 2:
        input_data_.lower_bounds = {0.0};
        input_data_.upper_bounds = {1.0};
        input_data_.num_steps = 100;
        input_data_.func = [](const std::vector<double> &pt) { return pt[0]; };
        expected_result_ = 0.5;
        break;
      case 3:
        input_data_.lower_bounds = {0.0};
        input_data_.upper_bounds = {1.0};
        input_data_.num_steps = 500;
        input_data_.func = [](const std::vector<double> &pt) { return pt[0] * pt[0]; };
        expected_result_ = 1.0 / 3.0;
        break;
      case 4:
        input_data_.lower_bounds = {0.0, 0.0};
        input_data_.upper_bounds = {1.0, 1.0};
        input_data_.num_steps = 100;
        input_data_.func = [](const std::vector<double> &pt) { return pt[0] + pt[1]; };
        expected_result_ = 1.0;
        break;
      case 5:
        input_data_.lower_bounds = {0.0, 0.0};
        input_data_.upper_bounds = {1.0, 1.0};
        input_data_.num_steps = 100;
        input_data_.func = [](const std::vector<double> &pt) { return pt[0] * pt[1]; };
        expected_result_ = 0.25;
        break;
      case 6:
        input_data_.lower_bounds = {0.0, 0.0, 0.0};
        input_data_.upper_bounds = {1.0, 1.0, 1.0};
        input_data_.num_steps = 30;
        input_data_.func = [](const std::vector<double> &pt) { return pt[0] + pt[1] + pt[2]; };
        expected_result_ = 1.5;
        break;
      case 7:
        input_data_.lower_bounds = {0.0, 0.0};
        input_data_.upper_bounds = {std::numbers::pi / 2, std::numbers::pi / 2};
        input_data_.num_steps = 100;
        input_data_.func = [](const std::vector<double> &pt) { return std::sin(pt[0]) * std::cos(pt[1]); };
        expected_result_ = 1.0;
        break;
      case 8:
        input_data_.lower_bounds = {0.0};
        input_data_.upper_bounds = {1.0};
        input_data_.num_steps = 100;
        input_data_.func = [](const std::vector<double> &pt) { return std::exp(-pt[0]); };
        expected_result_ = 1.0 - std::exp(-1.0);
        break;
      case 9:
        input_data_.lower_bounds = {0.0, 0.0};
        input_data_.upper_bounds = {2.0, 3.0};
        input_data_.num_steps = 50;
        input_data_.func = [](const std::vector<double> &) { return 1.0; };
        expected_result_ = 6.0;
        break;
      case 10:
        input_data_.lower_bounds = {0.0};
        input_data_.upper_bounds = {1.0};
        input_data_.num_steps = 1;
        input_data_.func = [](const std::vector<double> &) { return 1.0; };
        expected_result_ = 1.0;
        break;
      case 11:
        input_data_.lower_bounds = {1.0, 2.0};
        input_data_.upper_bounds = {3.0, 4.0};
        input_data_.num_steps = 100;
        input_data_.func = [](const std::vector<double> &) { return 1.0; };
        expected_result_ = 4.0;
        break;
      case 12:
        input_data_.lower_bounds = {0.0};
        input_data_.upper_bounds = {std::numbers::pi};
        input_data_.num_steps = 200;
        input_data_.func = [](const std::vector<double> &pt) { return std::sin(pt[0]); };
        expected_result_ = 2.0;
        break;
      case 13:
        input_data_.lower_bounds = {0.0, 0.0};
        input_data_.upper_bounds = {1.0, 1.0};
        input_data_.num_steps = 100;
        input_data_.func = [](const std::vector<double> &pt) { return std::exp(pt[0] + pt[1]); };
        expected_result_ = (std::numbers::e - 1.0) * (std::numbers::e - 1.0);
        break;
      case 14:
        input_data_.lower_bounds = {-1.0};
        input_data_.upper_bounds = {1.0};
        input_data_.num_steps = 200;
        input_data_.func = [](const std::vector<double> &pt) { return pt[0] * pt[0] * pt[0]; };
        expected_result_ = 0.0;
        break;
      case 15:
        input_data_.lower_bounds = {0.0, 0.0, 0.0};
        input_data_.upper_bounds = {1.0, 1.0, 1.0};
        input_data_.num_steps = 20;
        input_data_.func = [](const std::vector<double> &pt) { return pt[0] * pt[1] * pt[2]; };
        expected_result_ = 0.125;
        break;
      default:
        input_data_.lower_bounds = {0.0};
        input_data_.upper_bounds = {1.0};
        input_data_.num_steps = 100;
        input_data_.func = [](const std::vector<double> &) { return 1.0; };
        expected_result_ = 1.0;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_result_) < 1e-2;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double expected_result_ = 0.0;
};

namespace {

TEST_P(TochilinEIntegralTrapeziumFuncTests, IntegralTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 15> kTestParam = {
    std::make_tuple(1, "Constant1D"),    std::make_tuple(2, "Linear1D"),  std::make_tuple(3, "Quadratic1D"),
    std::make_tuple(4, "Sum2D"),         std::make_tuple(5, "Product2D"), std::make_tuple(6, "Sum3D"),
    std::make_tuple(7, "SinCos2D"),      std::make_tuple(8, "Exp1D"),     std::make_tuple(9, "Constant2DRect"),
    std::make_tuple(10, "SingleStep1D"), std::make_tuple(11, "Offset2D"), std::make_tuple(12, "SinPi1D"),
    std::make_tuple(13, "Exp2D"),        std::make_tuple(14, "Cubic1D"),  std::make_tuple(15, "Product3D")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TochilinEIntegralTrapeziumMPI, InType>(
                                               kTestParam, PPC_SETTINGS_tochilin_e_integral_trapezium),
                                           ppc::util::AddFuncTask<TochilinEIntegralTrapeziumSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_tochilin_e_integral_trapezium));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TochilinEIntegralTrapeziumFuncTests::PrintFuncTestName<TochilinEIntegralTrapeziumFuncTests>;

INSTANTIATE_TEST_SUITE_P(IntegralTests, TochilinEIntegralTrapeziumFuncTests, kGtestValues, kPerfTestName);

class TochilinEIntegralTrapeziumValidationTests : public ::testing::Test {};

TEST_F(TochilinEIntegralTrapeziumValidationTests, EmptyBoundsSeq) {
  InType input;
  input.lower_bounds = {};
  input.upper_bounds = {};
  input.num_steps = 100;
  input.func = [](const std::vector<double> &) { return 1.0; };
  auto task = std::make_shared<TochilinEIntegralTrapeziumSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

TEST_F(TochilinEIntegralTrapeziumValidationTests, MismatchedBoundsSeq) {
  InType input;
  input.lower_bounds = {0.0, 0.0};
  input.upper_bounds = {1.0};
  input.num_steps = 100;
  input.func = [](const std::vector<double> &) { return 1.0; };
  auto task = std::make_shared<TochilinEIntegralTrapeziumSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

TEST_F(TochilinEIntegralTrapeziumValidationTests, ZeroStepsSeq) {
  InType input;
  input.lower_bounds = {0.0};
  input.upper_bounds = {1.0};
  input.num_steps = 0;
  input.func = [](const std::vector<double> &) { return 1.0; };
  auto task = std::make_shared<TochilinEIntegralTrapeziumSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

TEST_F(TochilinEIntegralTrapeziumValidationTests, NegativeStepsSeq) {
  InType input;
  input.lower_bounds = {0.0};
  input.upper_bounds = {1.0};
  input.num_steps = -5;
  input.func = [](const std::vector<double> &) { return 1.0; };
  auto task = std::make_shared<TochilinEIntegralTrapeziumSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

TEST_F(TochilinEIntegralTrapeziumValidationTests, NullFuncSeq) {
  InType input;
  input.lower_bounds = {0.0};
  input.upper_bounds = {1.0};
  input.num_steps = 100;
  input.func = nullptr;
  auto task = std::make_shared<TochilinEIntegralTrapeziumSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

TEST_F(TochilinEIntegralTrapeziumValidationTests, InvertedBoundsSeq) {
  InType input;
  input.lower_bounds = {1.0};
  input.upper_bounds = {0.0};
  input.num_steps = 100;
  input.func = [](const std::vector<double> &) { return 1.0; };
  auto task = std::make_shared<TochilinEIntegralTrapeziumSEQ>(input);
  ASSERT_FALSE(task->Validation());
}

}  // namespace

}  // namespace tochilin_e_integral_trapezium
