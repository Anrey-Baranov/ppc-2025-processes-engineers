#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "bortsova_a_transmission_gather/common/include/common.hpp"
#include "bortsova_a_transmission_gather/mpi/include/ops_mpi.hpp"
#include "bortsova_a_transmission_gather/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace bortsova_a_transmission_gather {

namespace {

void FillWithSequence(std::vector<double> &vec, double start_val) {
  for (std::size_t idx = 0; idx < vec.size(); ++idx) {
    vec[idx] = start_val + static_cast<double>(idx);
  }
}

}  // namespace

class BortsovaATransmissionGatherFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int count = std::get<0>(params);
    input_.send_data.resize(static_cast<std::size_t>(count));
    FillWithSequence(input_.send_data, 1.0);
    input_.root = 0;

    std::string test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    is_mpi_test_ = test_name.find("_mpi_") != std::string::npos;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.recv_data.empty() && !input_.send_data.empty()) {
      return false;
    }

    int world_size = 1;
    if (is_mpi_test_ && ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    std::size_t expected_size = input_.send_data.size() * static_cast<std::size_t>(world_size);
    if (output_data.recv_data.size() != expected_size) {
      return false;
    }

    for (int rank = 0; rank < world_size; ++rank) {
      std::size_t offset = static_cast<std::size_t>(rank) * input_.send_data.size();
      for (std::size_t ii = 0; ii < input_.send_data.size(); ++ii) {
        if (output_data.recv_data[offset + ii] != input_.send_data[ii]) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
  bool is_mpi_test_ = false;
};

namespace {

TEST_P(BortsovaATransmissionGatherFuncTests, GatherTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BortsovaATransmissionGatherMPI, InType>(
                                               kTestParam, PPC_SETTINGS_bortsova_a_transmission_gather),
                                           ppc::util::AddFuncTask<BortsovaATransmissionGatherSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_bortsova_a_transmission_gather));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BortsovaATransmissionGatherFuncTests::PrintFuncTestName<BortsovaATransmissionGatherFuncTests>;

INSTANTIATE_TEST_SUITE_P(GatherTests, BortsovaATransmissionGatherFuncTests, kGtestValues, kPerfTestName);

}  // namespace

class BortsovaAGatherEdgeCasesTestsMPI : public ::testing::Test {
 protected:
  static void RunMPIGatherTest(const InType &input, int expected_count_per_rank) {
    auto task = std::make_shared<BortsovaATransmissionGatherMPI>(input);
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());

    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const auto &result = task->GetOutput().recv_data;
    std::size_t expected_total =
        static_cast<std::size_t>(expected_count_per_rank) * static_cast<std::size_t>(world_size);
    ASSERT_EQ(result.size(), expected_total);

    for (int rank = 0; rank < world_size; ++rank) {
      std::size_t offset = static_cast<std::size_t>(rank) * static_cast<std::size_t>(expected_count_per_rank);
      for (std::size_t ii = 0; ii < input.send_data.size(); ++ii) {
        EXPECT_DOUBLE_EQ(result[offset + ii], input.send_data[ii]);
      }
    }
  }
};

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, SingleElementGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data = {42.0};
  input.root = 0;
  RunMPIGatherTest(input, 1);
}

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, LargeDataGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data.resize(1000);
  FillWithSequence(input.send_data, 1.0);
  input.root = 0;
  RunMPIGatherTest(input, 1000);
}

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, NegativeValuesGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
  input.root = 0;
  RunMPIGatherTest(input, 6);
}

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, FloatingPointPrecisionGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data = {0.1, 0.2, 0.3, 1e-10, 1e10};
  input.root = 0;
  RunMPIGatherTest(input, 5);
}

TEST_F(BortsovaAGatherEdgeCasesTestsMPI, TenElementsGather) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  InType input;
  input.send_data.resize(10);
  FillWithSequence(input.send_data, 1.0);
  input.root = 0;
  RunMPIGatherTest(input, 10);
}

class BortsovaAGatherEdgeCasesTestsSEQ : public ::testing::Test {
 protected:
  static void RunSEQGatherTest(const InType &input, const std::vector<double> &expected) {
    auto task = std::make_shared<BortsovaATransmissionGatherSEQ>(input);
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());

    const auto &result = task->GetOutput().recv_data;
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t ii = 0; ii < expected.size(); ++ii) {
      EXPECT_DOUBLE_EQ(result[ii], expected[ii]);
    }
  }
};

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, SingleElementGather) {
  InType input;
  input.send_data = {42.0};
  input.root = 0;
  RunSEQGatherTest(input, {42.0});
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, LargeDataGather) {
  InType input;
  input.send_data.resize(1000);
  FillWithSequence(input.send_data, 1.0);
  input.root = 0;

  std::vector<double> expected(1000);
  FillWithSequence(expected, 1.0);
  RunSEQGatherTest(input, expected);
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, NegativeValuesGather) {
  InType input;
  input.send_data = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
  input.root = 0;
  RunSEQGatherTest(input, input.send_data);
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, FloatingPointPrecisionGather) {
  InType input;
  input.send_data = {0.1, 0.2, 0.3, 1e-10, 1e10};
  input.root = 0;
  RunSEQGatherTest(input, input.send_data);
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, EmptyDataGather) {
  InType input;
  input.send_data = {};
  input.root = 0;
  RunSEQGatherTest(input, {});
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, TwoElementsGather) {
  InType input;
  input.send_data = {1.5, 2.5};
  input.root = 0;
  RunSEQGatherTest(input, {1.5, 2.5});
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, IdenticalValuesGather) {
  InType input;
  input.send_data = {7.0, 7.0, 7.0, 7.0, 7.0};
  input.root = 0;
  RunSEQGatherTest(input, input.send_data);
}

TEST_F(BortsovaAGatherEdgeCasesTestsSEQ, AlternatingSignsGather) {
  InType input;
  input.send_data = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
  input.root = 0;
  RunSEQGatherTest(input, input.send_data);
}

}  // namespace bortsova_a_transmission_gather
