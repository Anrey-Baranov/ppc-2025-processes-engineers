#if 0  // NOLINT(readability-avoid-unconditional-preprocessor-if)
// Disabled because this is a higher-numbered duplicate (task 3) and conflicts with task 2 tests.
#  include <gtest/gtest.h>
#  include <mpi.h>

#  include <cstddef>
#  include <cstdint>
#  include <random>
#  include <stdexcept>
#  include <vector>

#  include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"
#  include "kondrashova_v_gauss_filter_vertical_split/mpi/include/ops_mpi.hpp"
#  include "kondrashova_v_gauss_filter_vertical_split/seq/include/ops_seq.hpp"
#  include "util/include/perf_test_util.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

class KondrashovaVRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static const int kWidth = 3840;
  static const int kHeight = 2160;
  static const int kChannels = 3;

  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_.width = kWidth;
    input_data_.height = kHeight;
    input_data_.channels = kChannels;
    input_data_.pixels.resize(static_cast<size_t>(kWidth) * kHeight * kChannels);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned int seed = 0;
    if (rank == 0) {
      std::random_device rd;
      seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);

    for (auto &pixel : input_data_.pixels) {
      pixel = static_cast<uint8_t>(dist(gen));
    }

    KondrashovaVGaussFilterVerticalSplitSEQ seq_task(input_data_);
    if (!seq_task.Validation() || !seq_task.PreProcessing() || !seq_task.Run() || !seq_task.PostProcessing()) {
      throw std::runtime_error("Failed to compute reference result");
    }
    expected_output_ = seq_task.GetOutput();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KondrashovaVRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KondrashovaVGaussFilterVerticalSplitMPI,
                                                       KondrashovaVGaussFilterVerticalSplitSEQ>(
    PPC_SETTINGS_kondrashova_v_gauss_filter_vertical_split);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KondrashovaVRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KondrashovaVRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kondrashova_v_gauss_filter_vertical_split
#endif
