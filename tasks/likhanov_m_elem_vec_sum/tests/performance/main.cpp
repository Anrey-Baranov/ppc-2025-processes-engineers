#include <gtest/gtest.h>
#include <mpi.h>

#include "likhanov_m_elem_vec_sum/common/include/common.hpp"
#include "likhanov_m_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "likhanov_m_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace likhanov_m_elem_vec_sum {

class LikhanovMElemVecSumRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr InType kCount = 100'000'000;
  InType input_data{};

  void SetUp() override {
    input_data = kCount;
  }

  InType GetTestInputData() final {
    return input_data;
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

    OutType expected = input_data * (input_data + 1) / 2;

    return output_data == expected;
  }
};

TEST_P(LikhanovMElemVecSumRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LikhanovMElemVecSumMPI, LikhanovMElemVecSumSEQ>(
    PPC_SETTINGS_likhanov_m_elem_vec_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunModeTests, LikhanovMElemVecSumRunPerfTests, kGtestValues,
                         LikhanovMElemVecSumRunPerfTests::CustomPerfTestName);

}  // namespace likhanov_m_elem_vec_sum
