#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "baranov_a_dijkstra_crs/mpi/include/ops_mpi.hpp"
#include "baranov_a_dijkstra_crs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace baranov_a_dijkstra_crs {

class BaranovADijkstraCrsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    auto param = GetParam();
    std::string task_name = std::get<1>(param);
    is_mpi_test_ = (task_name.find("mpi") != std::string::npos);
    int vertices = 10;

    GraphCRS graph;
    graph.vertices = vertices;
    graph.source = 0;
    graph.row_ptr.push_back(0);
    for (int i = 0; i < vertices; ++i) {
      if (i < vertices - 1) {
        graph.col_idx.push_back(i + 1);
      }
      graph.row_ptr.push_back(graph.col_idx.size());
    }
    graph.weights = 1;

    input_data_ = graph;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    try {
      auto output_vec = std::get<std::vector<double>>(output_data);

      if (output_vec.empty()) {
        return false;
      }
      return output_vec[0] == 0.0;

    } catch (const std::exception &) {
      return false;
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  bool is_mpi_test_ = false;
};

TEST_P(BaranovADijkstraCrsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BaranovADijkstraCrsMPI, BaranovADijkstraCrsSEQ>(
    PPC_SETTINGS_baranov_a_dijkstra_crs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BaranovADijkstraCrsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BaranovADijkstraCrsPerfTests, kGtestValues, kPerfTestName);

}  // namespace baranov_a_dijkstra_crs
