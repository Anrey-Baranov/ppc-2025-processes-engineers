#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <vector>

#include "alekseev_a_custom_reduce/common/include/common.hpp"
#include "alekseev_a_custom_reduce/mpi/include/ops_mpi.hpp"
#include "alekseev_a_custom_reduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace alekseev_a_custom_reduce {

class AlekseevACustomReduceRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    constexpr int kDataSize = 10'000'000;
    constexpr double kMultiplier = 0.3;

    input_data_.root = 0;
    input_data_.data.resize(kDataSize);

    expected_ = 0.0;

    for (int i = 0; i < kDataSize; ++i) {
      double value = std::fmod(static_cast<double>(i) * kMultiplier, 10000.0);
      input_data_.data[i] = value;
      expected_ += value;
    }
  }

  bool CheckTestOutputData(OutType &output_data) override {
    int initialized = 0;
    const int mpi_ok = MPI_Initialized(&initialized);

    int rank = 0;
    if ((mpi_ok == MPI_SUCCESS) && (initialized != 0)) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    constexpr double kAbsoluteEps = 1e-6;
    constexpr double kRelativeEps = 1e-12;

    const double difference = std::fabs(output_data - expected_);
    const double scale = std::fabs(expected_);
    const double threshold = kAbsoluteEps + (kRelativeEps * scale);
    if (rank == 0) {
      return difference <= threshold;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_{0.0};
};

TEST_P(AlekseevACustomReduceRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, AlekseevACustomReduceMPI, AlekseevACustomReduceSEQ>(
    PPC_SETTINGS_alekseev_a_custom_reduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = AlekseevACustomReduceRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, AlekseevACustomReduceRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace alekseev_a_custom_reduce
