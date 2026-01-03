#include <gtest/gtest.h>

#include <cstddef>

#include "borunov_v_block_partitioning/common/include/common.hpp"
#include "borunov_v_block_partitioning/mpi/include/ops_mpi.hpp"
#include "borunov_v_block_partitioning/seq/include/ops_seq.hpp"
#include "performance/include/performance.hpp"
#include "util/include/perf_test_util.hpp"

namespace borunov_v_block_partitioning {

class BorunovVBlockPartitioningPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int width = 4000;
    int height = 4000;

    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    input_data_.resize(static_cast<std::size_t>(2) + pixels);
    input_data_[0] = width;
    input_data_[1] = height;

    const int fill_value = 42;
    for (int i = 0; i < width * height; ++i) {
      input_data_[2 + i] = fill_value;
    }

    const std::size_t out_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    output_data_.resize(out_size);
  }

  void SetPerfAttributes(ppc::performance::PerfAttr &perf_attrs) override {
    ppc::util::BaseRunPerfTests<InType, OutType>::SetPerfAttributes(perf_attrs);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto width = input_data_[0];
    const auto height = input_data_[1];
    const std::size_t expected_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    return output_data.size() == expected_size;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_;
};

TEST_P(BorunovVBlockPartitioningPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BorunovVBlockPartitioningMPI, BorunovVBlockPartitioningSEQ>(
        PPC_SETTINGS_borunov_v_block_partitioning);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BorunovVBlockPartitioningPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BorunovVBlockPartitioningPerfTest, kGtestValues, kPerfTestName);

}  // namespace borunov_v_block_partitioning
