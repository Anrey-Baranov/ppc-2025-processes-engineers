#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "alekseev_a_global_opt_chars/common/include/common.hpp"
#include "alekseev_a_global_opt_chars/mpi/include/ops_mpi.hpp"
#include "alekseev_a_global_opt_chars/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace alekseev_a_global_opt_chars {

namespace {

double HimmelblauFunc(double x, double y) {
  const double t1 = ((x * x) + y - 11.0);
  const double t2 = (x + (y * y) - 7.0);
  return (t1 * t1) + (t2 * t2);
}

double RastriginFunc(double x, double y) {
  constexpr double kA = 10.0;
  constexpr double kTwoPi = 6.2831853071795864769;
  return (2.0 * kA) + ((x * x) - (kA * std::cos(kTwoPi * x))) + ((y * y) - (kA * std::cos(kTwoPi * y)));
}

double SphereFunc(double x, double y) {
  return (x * x) + (y * y);
}

double MatyasFunc(double x, double y) {
  return (0.26 * ((x * x) + (y * y))) - (0.48 * x * y);
}

double SimpleQuadratic(double x, double y) {
  return ((x - 2.0) * (x - 2.0)) + ((y - 3.0) * (y - 3.0));
}

double BoothFunc(double x, double y) {
  const double t1 = x + (2.0 * y) - 7.0;
  const double t2 = (2.0 * x) + y - 5.0;
  return (t1 * t1) + (t2 * t2);
}

}  // namespace

class AlekseevAGlobalOptCharsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int test_id = std::get<0>(params);

    input_data_ = InType{};

    switch (test_id) {
      case 1:
        input_data_.func = SphereFunc;
        input_data_.x_min = -2.0;
        input_data_.x_max = 2.0;
        input_data_.y_min = -2.0;
        input_data_.y_max = 2.0;
        break;

      case 2:
        input_data_.func = SimpleQuadratic;
        input_data_.x_min = 1.0;
        input_data_.x_max = 3.0;
        input_data_.y_min = 2.0;
        input_data_.y_max = 4.0;
        break;

      case 3:
        input_data_.func = MatyasFunc;
        input_data_.x_min = -3.0;
        input_data_.x_max = 3.0;
        input_data_.y_min = -3.0;
        input_data_.y_max = 3.0;
        break;

      case 4:
        input_data_.func = BoothFunc;
        input_data_.x_min = -10.0;
        input_data_.x_max = 10.0;
        input_data_.y_min = -10.0;
        input_data_.y_max = 10.0;
        break;

      case 5:
        input_data_.func = HimmelblauFunc;
        input_data_.x_min = -6.0;
        input_data_.x_max = 6.0;
        input_data_.y_min = -6.0;
        input_data_.y_max = 6.0;
        break;

      case 6:
        input_data_.func = RastriginFunc;
        input_data_.x_min = -5.12;
        input_data_.x_max = 5.12;
        input_data_.y_min = -5.12;
        input_data_.y_max = 5.12;
        break;

      default:
        FAIL();
    }

    input_data_.epsilon = 0.1;
    input_data_.r_param = 2.5;
    input_data_.max_iterations = 15;
  }

  bool CheckTestOutputData(OutType &output) final {
    return output.iterations >= 0 && output.x_opt >= input_data_.x_min && output.x_opt <= input_data_.x_max &&
           output.y_opt >= input_data_.y_min && output.y_opt <= input_data_.y_max;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(AlekseevAGlobalOptCharsFuncTests, GlobalOptimization) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(1, "Sphere"),     std::make_tuple(2, "Quadratic"),
                                            std::make_tuple(3, "Matyas"),     std::make_tuple(4, "Booth"),
                                            std::make_tuple(5, "Himmelblau"), std::make_tuple(6, "Rastrigin")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<AlekseevAGlobalOptCharsMPI, InType>(kTestParam, PPC_SETTINGS_alekseev_a_global_opt_chars),
    ppc::util::AddFuncTask<AlekseevAGlobalOptCharsSEQ, InType>(kTestParam, PPC_SETTINGS_alekseev_a_global_opt_chars));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AlekseevAGlobalOptCharsFuncTests::PrintFuncTestName<AlekseevAGlobalOptCharsFuncTests>;

INSTANTIATE_TEST_SUITE_P(OptimizationTests, AlekseevAGlobalOptCharsFuncTests, kGtestValues, kPerfTestName);

}  // namespace alekseev_a_global_opt_chars
