#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "baranov_a_dijkstra_crs/mpi/include/ops_mpi.hpp"
#include "baranov_a_dijkstra_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace baranov_a_dijkstra_crs {

class BaranovADijkstraCRSFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto param = GetParam();
    int test_case = std::get<0>(std::get<2>(param));

    switch (test_case) {
      case 1: {
        GraphData graph;
        graph.num_vertices = 4;
        graph.source_vertex = 0;

        graph.offsets = {0, 2, 4, 6, 6};
        graph.columns = {1, 2, 2, 3, 1, 3};
        graph.values = {10.0, 5.0, 2.0, 1.0, 3.0, 9.0};

        input_data_ = graph;
        expected_output_ = {0.0, 8.0, 5.0, 9.0};
        break;
      }
      case 2: {
        GraphData graph;
        graph.num_vertices = 4;
        graph.source_vertex = 0;

        graph.offsets = {0, 0, 0, 0, 0};
        graph.columns = {};
        graph.values = {};

        input_data_ = graph;
        expected_output_ = {0.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity()};
        break;
      }
      case 3: {
        GraphData graph;
        graph.num_vertices = 5;
        graph.source_vertex = 0;

        graph.offsets = {0, 1, 2, 3, 4, 4};
        graph.columns = {1, 2, 3, 4};
        graph.values = {1.0, 2.0, 3.0, 4.0};

        input_data_ = graph;
        expected_output_ = {0.0, 1.0, 3.0, 6.0, 10.0};
        break;
      }
      case 4: {
        GraphData graph;
        graph.num_vertices = 3;
        graph.source_vertex = 0;

        graph.offsets = {0, 2, 4, 6};
        graph.columns = {1, 2, 0, 2, 0, 1};
        graph.values = {1.0, 4.0, 1.0, 2.0, 4.0, 2.0};

        input_data_ = graph;
        expected_output_ = {0.0, 1.0, 3.0};
        break;
      }
      case 5: {
        GraphData graph;
        graph.num_vertices = 1;
        graph.source_vertex = 0;

        graph.offsets = {0, 0};
        graph.columns = {};
        graph.values = {};

        input_data_ = graph;
        expected_output_ = {0.0};
        break;
      }
      case 6: {
        GraphData graph;
        graph.num_vertices = 3;
        graph.source_vertex = 0;

        graph.offsets = {0, 2, 5, 6};
        graph.columns = {1, 2, 0, 2, 0, 1};
        graph.values = {5.0, 3.0, 5.0, 1.0, 3.0, 1.0};

        input_data_ = graph;
        expected_output_ = {0.0, 4.0, 3.0};
        break;
      }
      case 7: {
        GraphData graph;
        graph.num_vertices = 4;
        graph.source_vertex = 0;

        graph.offsets = {0, 2, 3, 4, 4};
        graph.columns = {1, 2, 3, 3};
        graph.values = {5.0, 3.0, 1.0, 4.0};

        input_data_ = graph;
        expected_output_ = {0.0, 5.0, 3.0, 6.0};
        break;
      }
      case 8: {
        GraphData graph;
        graph.num_vertices = 5;
        graph.source_vertex = 0;

        graph.offsets = {0, 4, 4, 4, 4, 4};
        graph.columns = {1, 2, 3, 4};
        graph.values = {1.0, 2.0, 3.0, 4.0};

        input_data_ = graph;
        expected_output_ = {0.0, 1.0, 2.0, 3.0, 4.0};
        break;
      }
      case 9: {
        GraphData graph;
        graph.num_vertices = 4;
        graph.source_vertex = 0;

        graph.offsets = {0, 1, 2, 3, 4};
        graph.columns = {1, 2, 3, 0};
        graph.values = {1.0, 1.0, 1.0, 1.0};

        input_data_ = graph;
        expected_output_ = {0.0, 1.0, 2.0, 3.0};
        break;
      }
      case 10: {
        GraphData graph;
        graph.num_vertices = 4;
        graph.source_vertex = 0;

        graph.offsets = {0, 2, 3, 4, 4};
        graph.columns = {1, 2, 3, 3};
        graph.values = {2.0, 3.0, 4.0, 1.0};

        input_data_ = graph;
        expected_output_ = {0.0, 2.0, 3.0, 4.0};
        break;
      }
      case 11: {
        GraphData graph;
        graph.num_vertices = 3;
        graph.source_vertex = 1;

        graph.offsets = {0, 0, 2, 4};
        graph.columns = {0, 2, 0, 2};
        graph.values = {2.0, 3.0, 2.0, 3.0};

        input_data_ = graph;
        expected_output_ = {2.0, 0.0, 3.0};
        break;
      }
      case 12: {
        GraphData graph;
        graph.num_vertices = 5;
        graph.source_vertex = 0;

        graph.offsets = {0, 2, 3, 4, 5, 5};
        graph.columns = {1, 2, 3, 4, 4};
        graph.values = {2.0, 4.0, 1.0, 3.0, 2.0};
        input_data_ = graph;
        expected_output_ = {0.0, 2.0, 4.0, 3.0, 5.0};
        break;
      }
      case 13: {
        GraphData graph;
        graph.num_vertices = 3;
        graph.source_vertex = 0;
        graph.offsets = {0, 2, 3, 3};
        graph.columns = {1, 2, 2};
        graph.values = {4.0, 2.0, 1.0};

        input_data_ = graph;
        expected_output_ = {0.0, 4.0, 2.0};
        break;
      }
      case 14: {
        GraphData graph;
        graph.num_vertices = 3;
        graph.source_vertex = 0;

        graph.offsets = {0, 2, 4, 6};
        graph.columns = {1, 2, 0, 2, 0, 1};
        graph.values = {1.0, 3.0, 1.0, 2.0, 3.0, 2.0};

        input_data_ = graph;
        expected_output_ = {0.0, 1.0, 3.0};
        break;
      }
      default:
        GraphData graph;
        graph.num_vertices = 4;
        graph.source_vertex = 0;
        graph.offsets = {0, 2, 4, 6, 6};
        graph.columns = {1, 2, 2, 3, 1, 3};
        graph.values = {10.0, 5.0, 2.0, 1.0, 3.0, 9.0};

        input_data_ = graph;
        expected_output_ = {0.0, 8.0, 5.0, 9.0};
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      bool expected_inf = std::isinf(expected_output_[i]);
      bool actual_inf = std::isinf(output_data[i]);

      if (expected_inf && actual_inf) {
        continue;
      }

      if (expected_inf != actual_inf) {
        return false;
      }

      if (std::abs(expected_output_[i] - output_data[i]) > 1e-6) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(BaranovADijkstraCRSFuncTests, DijkstraCRSTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {
    std::make_tuple(1, "simple_graph_4_vertices"),
    std::make_tuple(2, "no_paths_graph"),
    std::make_tuple(3, "linear_graph"),
    std::make_tuple(4, "complete_graph_3_vertices"),
    std::make_tuple(5, "single_node_graph"),
    std::make_tuple(6, "bidirectional_graph"),
    std::make_tuple(7, "graph_with_multiple_paths"),
    std::make_tuple(8, "star_graph"),
    std::make_tuple(9, "cyclic_graph"),
    std::make_tuple(10, "multiple_paths_to_target"),
    std::make_tuple(11, "non_zero_source"),
    std::make_tuple(12, "large_sparse_graph"),
    std::make_tuple(13, "graph_with_self_loops"),
    std::make_tuple(14, "fully_connected_graph"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BaranovADijkstraCRSMPI, InType>(kTestParam, PPC_SETTINGS_baranov_a_dijkstra_crs),
    ppc::util::AddFuncTask<BaranovADijkstraCRSSEQ, InType>(kTestParam, PPC_SETTINGS_baranov_a_dijkstra_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaranovADijkstraCRSFuncTests::PrintFuncTestName<BaranovADijkstraCRSFuncTests>;

INSTANTIATE_TEST_SUITE_P(DijkstraCRSTests, BaranovADijkstraCRSFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baranov_a_dijkstra_crs
