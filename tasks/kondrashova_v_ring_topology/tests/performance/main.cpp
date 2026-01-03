#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "kondrashova_v_ring_topology/common/include/common.hpp"
#include "kondrashova_v_ring_topology/mpi/include/ops_mpi.hpp"
#include "kondrashova_v_ring_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kondrashova_v_ring_topology {

class KondrashovaVRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 100000000;

  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    input_data_.source = 0;
    input_data_.recipient = world_size - 1;

    input_data_.data.resize(kDataSize);
    for (int i = 0; i < kDataSize; ++i) {
      input_data_.data[i] = i;
    }

    expected_output_ = input_data_.data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == input_data_.recipient) {
      if (output_data.size() != expected_output_.size()) {
        return false;
      }
      return output_data == expected_output_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KondrashovaVRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KondrashovaVRingTopologyMPI, KondrashovaVRingTopologySEQ>(
        PPC_SETTINGS_kondrashova_v_ring_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KondrashovaVRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KondrashovaVRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kondrashova_v_ring_topology
