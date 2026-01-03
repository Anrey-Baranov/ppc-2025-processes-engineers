#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "dorogin_v_bitwise_simple_merge/common/include/common.hpp"
#include "dorogin_v_bitwise_simple_merge/mpi/include/ops_mpi.hpp"
#include "dorogin_v_bitwise_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dorogin_v_bitwise_simple_merge {

class DoroginVBitwiseSimpleMergeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);
    input_data_.resize(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (int i = 0; i < size; i++) {
      input_data_[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != input_data_.size()) {
      return false;
    }
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(DoroginVBitwiseSimpleMergeFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(100, "size100"), std::make_tuple(500, "size500"),
                                            std::make_tuple(1000, "size1000")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<DoroginVBitwiseSimpleMergeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_dorogin_v_bitwise_simple_merge),
                                           ppc::util::AddFuncTask<DoroginVBitwiseSimpleMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_dorogin_v_bitwise_simple_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = DoroginVBitwiseSimpleMergeFuncTests::PrintFuncTestName<DoroginVBitwiseSimpleMergeFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, DoroginVBitwiseSimpleMergeFuncTests, kGtestValues, kPerfTestName);

}  // namespace

class DoroginVBitwiseSimpleMergeValidationTests : public ::testing::Test {};

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, EmptyInput) {
  std::vector<double> input;
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  EXPECT_TRUE(task.GetOutput().empty());
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, SingleElement) {
  std::vector<double> input = {42.5};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  ASSERT_EQ(task.GetOutput().size(), 1U);
  EXPECT_DOUBLE_EQ(task.GetOutput()[0], 42.5);
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, TwoElements) {
  std::vector<double> input = {5.0, 2.0};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  std::vector<double> expected = {2.0, 5.0};
  ASSERT_EQ(task.GetOutput().size(), expected.size());
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], expected[i]);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, AlreadySorted) {
  std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], input[i]);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, ReverseSorted) {
  std::vector<double> input = {5.0, 4.0, 3.0, 2.0, 1.0};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  std::vector<double> expected = {1.0, 2.0, 3.0, 4.0, 5.0};
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], expected[i]);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, NegativeNumbers) {
  std::vector<double> input = {-3.0, -1.0, -5.0, -2.0, -4.0};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  std::vector<double> expected = {-5.0, -4.0, -3.0, -2.0, -1.0};
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], expected[i]);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, MixedPositiveNegative) {
  std::vector<double> input = {3.0, -1.0, 0.0, -5.0, 2.0};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  std::vector<double> expected = {-5.0, -1.0, 0.0, 2.0, 3.0};
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], expected[i]);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, DuplicateValues) {
  std::vector<double> input = {3.0, 1.0, 3.0, 2.0, 1.0};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  std::vector<double> expected = {1.0, 1.0, 2.0, 3.0, 3.0};
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], expected[i]);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, AllSameValues) {
  std::vector<double> input = {7.5, 7.5, 7.5, 7.5};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], 7.5);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, SmallFractionalValues) {
  std::vector<double> input = {0.003, 0.001, 0.002, 0.005, 0.004};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  std::vector<double> expected = {0.001, 0.002, 0.003, 0.004, 0.005};
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], expected[i]);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, LargeValues) {
  std::vector<double> input = {1e10, 1e8, 1e12, 1e6, 1e9};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  std::vector<double> expected = {1e6, 1e8, 1e9, 1e10, 1e12};
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], expected[i]);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, MpiEmptyInput) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  std::vector<double> input;
  DoroginVBitwiseSimpleMergeMPI task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, MpiSingleElement) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  std::vector<double> input = {42.5};
  DoroginVBitwiseSimpleMergeMPI task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, MpiMixedValues) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  std::vector<double> input = {5.0, -3.0, 0.0, 7.0, -1.0, 2.0};
  DoroginVBitwiseSimpleMergeMPI task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, RandomDataSeq) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-500.0, 500.0);
  std::vector<double> input(50);
  for (auto &v : input) {
    v = dist(gen);
  }

  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  EXPECT_TRUE(std::ranges::is_sorted(task.GetOutput()));
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, ZeroValues) {
  std::vector<double> input = {0.0, 0.0, 0.0};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], 0.0);
  }
}

TEST_F(DoroginVBitwiseSimpleMergeValidationTests, VerySmallNegativeAndPositive) {
  std::vector<double> input = {1e-15, -1e-15, 1e-14, -1e-14};
  DoroginVBitwiseSimpleMergeSEQ task(input);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  std::vector<double> expected = {-1e-14, -1e-15, 1e-15, 1e-14};
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_DOUBLE_EQ(task.GetOutput()[i], expected[i]);
  }
}

}  // namespace dorogin_v_bitwise_simple_merge
