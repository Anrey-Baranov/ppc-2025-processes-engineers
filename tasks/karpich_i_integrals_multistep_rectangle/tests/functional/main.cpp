#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "karpich_i_integrals_multistep_rectangle/common/include/common.hpp"
#include "karpich_i_integrals_multistep_rectangle/mpi/include/ops_mpi.hpp"
#include "karpich_i_integrals_multistep_rectangle/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace karpich_i_integrals_multistep_rectangle {

static double ComputeExpectedResult(int n) {
  double result = 0.0;
  for (int step = 1; step <= 3; step++) {
    int divisions = n * step;
    double h = 1.0 / divisions;
    double step_result = 0.0;

    for (int i = 0; i < divisions; i++) {
      for (int j = 0; j < divisions; j++) {
        for (int k = 0; k < divisions; k++) {
          double x = (i + 0.5) * h;
          double y = (j + 0.5) * h;
          double z = (k + 0.5) * h;
          step_result += (x * y * z + x * x + y * y + z * z) * h * h * h;
        }
      }
    }
    result += step_result;
  }
  return result / 3.0;
}

class KarpichIIntegralsMultistepRectangleFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_result_ = static_cast<OutType>(std::round(ComputeExpectedResult(input_data_)));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(static_cast<int>(output_data) - static_cast<int>(expected_result_)) <= 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
  OutType expected_result_ = 0;
};

namespace {

TEST_P(KarpichIIntegralsMultistepRectangleFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 20> kTestParam = {
    std::make_tuple(1, "1"),   std::make_tuple(2, "2"),   std::make_tuple(3, "3"),   std::make_tuple(4, "4"),
    std::make_tuple(5, "5"),   std::make_tuple(6, "6"),   std::make_tuple(7, "7"),   std::make_tuple(8, "8"),
    std::make_tuple(9, "9"),   std::make_tuple(10, "10"), std::make_tuple(11, "11"), std::make_tuple(12, "12"),
    std::make_tuple(13, "13"), std::make_tuple(14, "14"), std::make_tuple(15, "15"), std::make_tuple(16, "16"),
    std::make_tuple(17, "17"), std::make_tuple(18, "18"), std::make_tuple(19, "19"), std::make_tuple(20, "20")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KarpichIIntegralsMultistepRectangleMPI, InType>(
                                               kTestParam, PPC_SETTINGS_karpich_i_integrals_multistep_rectangle),
                                           ppc::util::AddFuncTask<KarpichIIntegralsMultistepRectangleSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_karpich_i_integrals_multistep_rectangle));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KarpichIIntegralsMultistepRectangleFuncTests::PrintFuncTestName<KarpichIIntegralsMultistepRectangleFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KarpichIIntegralsMultistepRectangleFuncTests, kGtestValues, kPerfTestName);

TEST(KarpichIIntegralsMultistepRectangleFuncTests, BoundaryCaseMinValue) {
  KarpichIIntegralsMultistepRectangleSEQ seq_task(1);
  EXPECT_TRUE(seq_task.Validation());
  EXPECT_TRUE(seq_task.PreProcessing());
  EXPECT_TRUE(seq_task.Run());
  EXPECT_TRUE(seq_task.PostProcessing());
  EXPECT_GT(seq_task.GetOutput(), 0);
}

TEST(KarpichIIntegralsMultistepRectangleFuncTests, BoundaryCaseSmallValue) {
  KarpichIIntegralsMultistepRectangleSEQ seq_task(2);
  EXPECT_TRUE(seq_task.Validation());
  EXPECT_TRUE(seq_task.PreProcessing());
  EXPECT_TRUE(seq_task.Run());
  EXPECT_TRUE(seq_task.PostProcessing());
  EXPECT_GT(seq_task.GetOutput(), 0);
}

TEST(KarpichIIntegralsMultistepRectangleFuncTests, BoundaryCaseMediumValue) {
  KarpichIIntegralsMultistepRectangleSEQ seq_task(15);
  EXPECT_TRUE(seq_task.Validation());
  EXPECT_TRUE(seq_task.PreProcessing());
  EXPECT_TRUE(seq_task.Run());
  EXPECT_TRUE(seq_task.PostProcessing());
  EXPECT_GT(seq_task.GetOutput(), 0);
}

TEST(KarpichIIntegralsMultistepRectangleFuncTests, BoundaryCaseLargeValue) {
  KarpichIIntegralsMultistepRectangleSEQ seq_task(30);
  EXPECT_TRUE(seq_task.Validation());
  EXPECT_TRUE(seq_task.PreProcessing());
  EXPECT_TRUE(seq_task.Run());
  EXPECT_TRUE(seq_task.PostProcessing());
  EXPECT_GT(seq_task.GetOutput(), 0);
}

TEST(KarpichIIntegralsMultistepRectangleFuncTests, ValidationTestPositive) {
  KarpichIIntegralsMultistepRectangleSEQ seq_task(5);
  EXPECT_TRUE(seq_task.Validation());
}

TEST(KarpichIIntegralsMultistepRectangleFuncTests, PreProcessingTest) {
  KarpichIIntegralsMultistepRectangleSEQ seq_task(5);
  EXPECT_TRUE(seq_task.PreProcessing());
  EXPECT_EQ(seq_task.GetOutput(), 0);
}

TEST(KarpichIIntegralsMultistepRectangleFuncTests, PostProcessingTest) {
  KarpichIIntegralsMultistepRectangleSEQ seq_task(5);
  EXPECT_TRUE(seq_task.PreProcessing());
  EXPECT_TRUE(seq_task.Run());
  OutType before_post = seq_task.GetOutput();
  EXPECT_TRUE(seq_task.PostProcessing());
  EXPECT_EQ(seq_task.GetOutput(), before_post);
}

TEST(KarpichIIntegralsMultistepRectangleFuncTests, ConsistencyTest) {
  KarpichIIntegralsMultistepRectangleSEQ seq_task1(10);
  KarpichIIntegralsMultistepRectangleSEQ seq_task2(10);

  seq_task1.PreProcessing();
  seq_task1.Run();
  seq_task1.PostProcessing();

  seq_task2.PreProcessing();
  seq_task2.Run();
  seq_task2.PostProcessing();

  EXPECT_EQ(seq_task1.GetOutput(), seq_task2.GetOutput());
}

}  // namespace

}  // namespace karpich_i_integrals_multistep_rectangle
