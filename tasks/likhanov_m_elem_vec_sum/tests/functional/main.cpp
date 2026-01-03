#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>

#include "likhanov_m_elem_vec_sum/common/include/common.hpp"
#include "likhanov_m_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "likhanov_m_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace likhanov_m_elem_vec_sum {

class LikhanovMElemVecSumRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "N_" + std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    TestType tt = std::get<static_cast<std::int64_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(tt);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    int rank = 0;
    if (mpi_initialized == 1) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    TestType tt = std::get<static_cast<std::int64_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    InType n = std::get<0>(tt);

    OutType expected = n * (n + 1) / 2;
    return output_data == expected;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(LikhanovMElemVecSumRunFuncTests, ElemVecSumFromN) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(0, 0),   std::make_tuple(1, 0),    std::make_tuple(10, 0),
    std::make_tuple(100, 0), std::make_tuple(1000, 0),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LikhanovMElemVecSumMPI, InType>(kTestParam, PPC_SETTINGS_likhanov_m_elem_vec_sum),
    ppc::util::AddFuncTask<LikhanovMElemVecSumSEQ, InType>(kTestParam, PPC_SETTINGS_likhanov_m_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = LikhanovMElemVecSumRunFuncTests::PrintFuncTestName<LikhanovMElemVecSumRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(ElemVecSumTests, LikhanovMElemVecSumRunFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace likhanov_m_elem_vec_sum
