#include <gtest/gtest.h>
#include <mpi.h>

#include "kolotukhin_a_hypercube/common/include/common.hpp"
#include "kolotukhin_a_hypercube/mpi/include/ops_mpi.hpp"
#include "kolotukhin_a_hypercube/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kolotukhin_a_hypercube {

class KolotukhinAHypercubePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  int kDataSize_ = 400;
  InType input_data_;
  void SetUp() override {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    input_data_.resize(3);
    input_data_[0] = 0;
    input_data_[1] = world_size - 1;
    input_data_[2] = kDataSize_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == kDataSize_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KolotukhinAHypercubePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KolotukhinAHypercubeMPI, KolotukhinAHypercubeSEQ>(
    PPC_SETTINGS_kolotukhin_a_hypercube);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KolotukhinAHypercubePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KolotukhinAHypercubePerfTests, kGtestValues, kPerfTestName);

}  // namespace kolotukhin_a_hypercube
