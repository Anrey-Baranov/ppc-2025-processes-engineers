#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "klimenko_v_multistep_2d_parallel_sad/common/include/common.hpp"
#include "klimenko_v_multistep_2d_parallel_sad/mpi/include/ops_mpi.hpp"
#include "klimenko_v_multistep_2d_parallel_sad/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace klimenko_v_multistep_2d_parallel_sad {

namespace {

double SphereFunc(double x, double y) {
  return (x * x) + (y * y);
}

double SimpleQuadratic(double x, double y) {
  return ((x - 2.0) * (x - 2.0)) + ((y - 3.0) * (y - 3.0));
}

double MatyasFunc(double x, double y) {
  return (0.26 * (x * x + y * y)) - (0.48 * x * y);
}

}  // namespace

class KlimenkoV2DParallelSadFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int test_id = std::get<0>(params);
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

TEST_P(KlimenkoV2DParallelSadFuncTests, GlobalOptimization) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "Sphere"), std::make_tuple(2, "Quadratic"),
                                            std::make_tuple(3, "Matyas")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KlimenkoV2DParallelSadMPI, InType>(
                                               kTestParam, PPC_SETTINGS_klimenko_v_multistep_2d_parallel_sad),
                                           ppc::util::AddFuncTask<KlimenkoV2DParallelSadSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_klimenko_v_multistep_2d_parallel_sad));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KlimenkoV2DParallelSadFuncTests::PrintFuncTestName<KlimenkoV2DParallelSadFuncTests>;

INSTANTIATE_TEST_SUITE_P(OptimizationTests, KlimenkoV2DParallelSadFuncTests, kGtestValues, kPerfTestName);

}  // namespace klimenko_v_multistep_2d_parallel_sad
