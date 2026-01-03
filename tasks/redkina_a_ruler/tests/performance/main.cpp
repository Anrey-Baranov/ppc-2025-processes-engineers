#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "redkina_a_ruler/common/include/common.hpp"
#include "redkina_a_ruler/mpi/include/ops_mpi.hpp"
#include "redkina_a_ruler/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace redkina_a_ruler {

class RedkinaARulerPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 100000000;
  InType input_data_{};

  void SetUp() override {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    input_data_.start = 0;
    input_data_.end = (size > 1) ? size - 1 : 0;
    input_data_.data.resize(kDataSize);
    for (std::size_t i = 0; i < input_data_.data.size(); ++i) {
      input_data_.data[i] = static_cast<int>(i);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == input_data_.data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RedkinaARulerPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RedkinaARulerMPI, RedkinaARulerSEQ>(PPC_SETTINGS_redkina_a_ruler);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RedkinaARulerPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RedkinaARulerPerfTest, kGtestValues, kPerfTestName);

}  // namespace redkina_a_ruler
