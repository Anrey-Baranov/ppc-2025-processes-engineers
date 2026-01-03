#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <thread>

#include "trofimov_n_linear_topology/common/include/common.hpp"
#include "trofimov_n_linear_topology/mpi/include/ops_mpi.hpp"
#include "trofimov_n_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace trofimov_n_linear_topology {

class TrofimovNLinearTopologyPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kValue_ = 100000;
  InType input_data_{};

  void SetUp() override {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    input_data_.source = 0;
    input_data_.target = world_size - 1;
    input_data_.value = kValue_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_.value == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TrofimovNLinearTopologyPerfTest, RunPerfModes) {
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TrofimovNLinearTopologyMPI, TrofimovNLinearTopologySEQ>(
    PPC_SETTINGS_trofimov_n_linear_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TrofimovNLinearTopologyPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TrofimovNLinearTopologyPerfTest, kGtestValues, kPerfTestName);

}  // namespace trofimov_n_linear_topology
