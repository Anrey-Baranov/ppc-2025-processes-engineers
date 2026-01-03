#include <gtest/gtest.h>

#include <array>
#include <cstddef>

#include "util/include/perf_test_util.hpp"
#include "yakimov_i_linear_virtual_topology/common/include/common.hpp"
#include "yakimov_i_linear_virtual_topology/mpi/include/ops_mpi.hpp"
#include "yakimov_i_linear_virtual_topology/seq/include/ops_seq.hpp"

namespace yakimov_i_linear_virtual_topology {

class YakimovILinearVirtualTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    static size_t test_index = 0;
    static constexpr std::array<InType, 4> kTestSizes = {27, 28, 29, 30};
    InType result = kTestSizes.at(test_index % kTestSizes.size());
    test_index++;
    return result;
  }
};

TEST_P(YakimovILinearVirtualTopologyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, YakimovILinearVirtualTopologyMPI, YakimovILinearVirtualTopologySEQ>(
        PPC_SETTINGS_yakimov_i_linear_virtual_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YakimovILinearVirtualTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YakimovILinearVirtualTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace yakimov_i_linear_virtual_topology
