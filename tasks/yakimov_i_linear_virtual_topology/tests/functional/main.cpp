#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yakimov_i_linear_virtual_topology/common/include/common.hpp"
#include "yakimov_i_linear_virtual_topology/mpi/include/ops_mpi.hpp"
#include "yakimov_i_linear_virtual_topology/seq/include/ops_seq.hpp"

namespace yakimov_i_linear_virtual_topology {

class YakimovILinearVirtualTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return std::get<0>(params);
  }
};

namespace {

TEST_P(YakimovILinearVirtualTopologyFuncTests, LinearVirtualTopology) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kAllTestParam = {
    std::make_tuple(1, "basic_1"),          std::make_tuple(2, "basic_2"),
    std::make_tuple(3, "basic_3"),          std::make_tuple(4, "basic_4"),
    std::make_tuple(5, "basic_5"),          std::make_tuple(32, "edge_invalid_process"),
    std::make_tuple(33, "edge_large_data"), std::make_tuple(34, "edge_negative_values"),
    std::make_tuple(35, "edge_mixed"),      std::make_tuple(36, "edge_max_processes")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<YakimovILinearVirtualTopologyMPI, InType>(
                                               kAllTestParam, PPC_SETTINGS_yakimov_i_linear_virtual_topology),
                                           ppc::util::AddFuncTask<YakimovILinearVirtualTopologySEQ, InType>(
                                               kAllTestParam, PPC_SETTINGS_yakimov_i_linear_virtual_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    YakimovILinearVirtualTopologyFuncTests::PrintFuncTestName<YakimovILinearVirtualTopologyFuncTests>;

INSTANTIATE_TEST_SUITE_P(TopologyTests, YakimovILinearVirtualTopologyFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace yakimov_i_linear_virtual_topology
