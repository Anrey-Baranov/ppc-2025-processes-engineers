#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "makoveeva_s_simple_iteration/common/include/common.hpp"
#include "makoveeva_s_simple_iteration/mpi/include/ops_mpi.hpp"
#include "makoveeva_s_simple_iteration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace makoveeva_s_simple_iteration {

class MakoveevaSSimpleIterationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data > 0 && output_data < input_data_ * input_data_ * 10;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(MakoveevaSSimpleIterationFuncTests, SimpleIterationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(1, "size_1"),   std::make_tuple(2, "size_2"),
                                             std::make_tuple(3, "size_3"),   std::make_tuple(5, "size_5"),
                                             std::make_tuple(7, "size_7"),   std::make_tuple(10, "size_10"),
                                             std::make_tuple(15, "size_15"), std::make_tuple(20, "size_20"),
                                             std::make_tuple(30, "size_30"), std::make_tuple(50, "size_50")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MakoveevaSSimpleIterationMPI, InType>(kTestParam, PPC_SETTINGS_makoveeva_s_simple_iteration),
    ppc::util::AddFuncTask<MakoveevaSSimpleIterationSEQ, InType>(kTestParam,
                                                                 PPC_SETTINGS_makoveeva_s_simple_iteration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = MakoveevaSSimpleIterationFuncTests::PrintFuncTestName<MakoveevaSSimpleIterationFuncTests>;

INSTANTIATE_TEST_SUITE_P(BasicTests, MakoveevaSSimpleIterationFuncTests, kGtestValues, kFuncTestName);

TEST(MakoveevaSSimpleIterationEdgeCases, InvalidInputZeroSEQ) {
  MakoveevaSSimpleIterationSEQ task(0);
  EXPECT_FALSE(task.Validation());
}

TEST(MakoveevaSSimpleIterationEdgeCases, InvalidInputNegativeSEQ) {
  MakoveevaSSimpleIterationSEQ task(-5);
  EXPECT_FALSE(task.Validation());
}

TEST(MakoveevaSSimpleIterationEdgeCases, ValidInputPositiveSEQ) {
  MakoveevaSSimpleIterationSEQ task(5);
  EXPECT_TRUE(task.Validation());
}

TEST(MakoveevaSSimpleIterationEdgeCases, PreProcessingSEQ) {
  MakoveevaSSimpleIterationSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(MakoveevaSSimpleIterationEdgeCases, FullExecutionSEQ) {
  MakoveevaSSimpleIterationSEQ task(5);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_GT(task.GetOutput(), 0);
}

TEST(MakoveevaSSimpleIterationEdgeCases, InvalidInputZeroMPI) {
  MakoveevaSSimpleIterationMPI task(0);
  EXPECT_FALSE(task.Validation());
}

TEST(MakoveevaSSimpleIterationEdgeCases, InvalidInputNegativeMPI) {
  MakoveevaSSimpleIterationMPI task(-5);
  EXPECT_FALSE(task.Validation());
}

TEST(MakoveevaSSimpleIterationEdgeCases, ValidInputPositiveMPI) {
  MakoveevaSSimpleIterationMPI task(5);
  EXPECT_TRUE(task.Validation());
}

}  // namespace

}  // namespace makoveeva_s_simple_iteration
