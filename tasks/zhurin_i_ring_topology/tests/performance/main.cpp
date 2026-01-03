#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zhurin_i_ring_topology/common/include/common.hpp"
#include "zhurin_i_ring_topology/mpi/include/ops_mpi.hpp"
#include "zhurin_i_ring_topology/seq/include/ops_seq.hpp"

namespace zhurin_i_ring_topology {

class ZhurinIRingTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    std::random_device rand;
    std::mt19937_64 rng(rand());

    int world_size = 0;  // для clang-tidy, но не  const int world_size = 8; как было в предыдущей реализации
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::uniform_int_distribution<int> data_dist(1, 100);

    input_data_.source = 0;

    input_data_.dest = std::min(4, world_size - 1);

    input_data_.go_clockwise = true;

    const size_t data_size = 100000000;
    input_data_.data.resize(data_size);

    for (size_t i = 0; i < data_size; ++i) {
      input_data_.data[i] = data_dist(rng);
    }

    expected_data_.clear();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }
    return true;
  }

  [[nodiscard]] InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_data_;
};

namespace {

TEST_P(ZhurinIRingTopologyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZhurinIRingTopologyMPI, ZhurinIRingTopologySEQ>(
    PPC_SETTINGS_zhurin_i_ring_topology);

inline const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

inline const auto kPerfTestName = ZhurinIRingTopologyPerfTests::CustomPerfTestName;

// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(ZhurinRingTopologyPerf, ZhurinIRingTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zhurin_i_ring_topology
