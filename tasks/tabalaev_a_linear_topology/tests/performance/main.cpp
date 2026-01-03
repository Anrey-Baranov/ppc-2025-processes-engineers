#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "tabalaev_a_linear_topology/common/include/common.hpp"
#include "tabalaev_a_linear_topology/mpi/include/ops_mpi.hpp"
#include "tabalaev_a_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tabalaev_a_linear_topology {

class TabalaevALinearTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    int sender = 0;
    int receiver = 1;

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int world_size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      receiver = world_size - 1;
    }

    size_t size = 15000000;

    std::vector<int> data(size);
    for (int i = 0; std::cmp_less(i, size); i++) {
      data[i] = (i * i) + 2;
    }

    input_data_ = std::make_tuple(sender, receiver, data);
    output_data_ = data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == output_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<int> output_data_;
};

TEST_P(TabalaevALinearTopologyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TabalaevALinearTopologyMPI, TabalaevALinearTopologySEQ>(
    PPC_SETTINGS_tabalaev_a_linear_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TabalaevALinearTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TabalaevALinearTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace tabalaev_a_linear_topology
