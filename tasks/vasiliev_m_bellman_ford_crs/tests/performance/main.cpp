#include <gtest/gtest.h>

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"
#include "vasiliev_m_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_bellman_ford_crs/seq/include/ops_seq.hpp"

namespace vasiliev_m_bellman_ford_crs {

class VasilievMBellmanFordCrsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    std::ifstream file(ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_bellman_ford_crs, "tree_perf_test.txt"));

    if (!file.is_open()) {
      throw std::runtime_error("Cannot open perf test file");
    }

    int num_vertices = 0;
    int num_edges = 0;
    file >> num_vertices >> num_edges;

    std::vector<std::tuple<int, int, int>> edges(num_edges);
    for (int i = 0; i < num_edges; i++) {
      file >> std::get<0>(edges[i]) >> std::get<1>(edges[i]) >> std::get<2>(edges[i]);
    }

    int source = 0;
    file >> source;

    input_data_.row_ptr.assign(num_vertices + 1, 0);
    for (const auto &e : edges) {
      input_data_.row_ptr[std::get<0>(e) + 1]++;
    }

    for (int i = 1; i <= num_vertices; i++) {
      input_data_.row_ptr[i] += input_data_.row_ptr[i - 1];
    }

    input_data_.col_ind.resize(num_edges);
    input_data_.vals.resize(num_edges);

    std::vector<int> counter = input_data_.row_ptr;
    for (const auto &e : edges) {
      int u = std::get<0>(e);
      int idx = counter[u]++;
      input_data_.col_ind[idx] = std::get<1>(e);
      input_data_.vals[idx] = std::get<2>(e);
    }

    input_data_.source = source;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(VasilievMBellmanFordCrsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VasilievMBellmanFordCrsMPI, VasilievMBellmanFordCrsSEQ>(
    PPC_SETTINGS_vasiliev_m_bellman_ford_crs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VasilievMBellmanFordCrsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VasilievMBellmanFordCrsPerfTests, kGtestValues, kPerfTestName);

}  // namespace vasiliev_m_bellman_ford_crs
