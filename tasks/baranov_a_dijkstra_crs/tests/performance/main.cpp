#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "baranov_a_dijkstra_crs/mpi/include/ops_mpi.hpp"
#include "baranov_a_dijkstra_crs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace baranov_a_dijkstra_crs {

class BaranovADijkstraCRSPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int num_vertices = 1000;
    int edges_per_vertex = 10;

    GraphData graph;
    graph.num_vertices = num_vertices;
    graph.source_vertex = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertex_dis(0, num_vertices - 1);
    std::uniform_real_distribution<> weight_dis(0.1, 10.0);

    graph.offsets.resize(num_vertices + 1);
    graph.offsets[0] = 0;

    for (int i = 0; i < num_vertices; ++i) {
      int num_edges = edges_per_vertex;
      graph.offsets[i + 1] = graph.offsets[i] + num_edges;

      for (int j = 0; j < num_edges; ++j) {
        int target = vertex_dis(gen);
        double weight = weight_dis(gen);

        graph.columns.push_back(target);
        graph.values.push_back(weight);
      }
    }

    input_data_ = graph;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(BaranovADijkstraCRSPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BaranovADijkstraCRSMPI, BaranovADijkstraCRSSEQ>(
    PPC_SETTINGS_baranov_a_dijkstra_crs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BaranovADijkstraCRSPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BaranovADijkstraCRSPerfTests, kGtestValues, kPerfTestName);

}  // namespace baranov_a_dijkstra_crs
