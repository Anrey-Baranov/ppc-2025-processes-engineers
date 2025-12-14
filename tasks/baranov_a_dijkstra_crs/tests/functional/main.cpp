#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "baranov_a_dijkstra_crs/mpi/include/ops_mpi.hpp"
#include "baranov_a_dijkstra_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace baranov_a_dijkstra_crs {

class BaranovADijkstraCrsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  BaranovADijkstraCrsFuncTests() {}

  void SetUp() override {
    auto param = GetParam();
    TestType test_param = std::get<2>(param);
    int test_case = std::get<0>(test_param);
    std::string task_name = std::get<1>(param);
    is_mpi_test_ = (task_name.find("mpi") != std::string::npos);

    switch (test_case) {
      case 1: {
        input_data_ = CreateSimpleGraph(5, 1, "int");
      } break;

      case 2: {
        input_data_ = CreateSimpleGraph(10, 2, "int");
      } break;

      case 3: {
        input_data_ = CreateSimpleGraph(8, 1.5, "float");
      } break;

      case 4: {
        input_data_ = CreateSimpleGraph(6, 2.7, "double");
      } break;

      case 5: {
        input_data_ = CreateSingleVertexGraph();
      } break;

      case 6: {
        input_data_ = CreateCompleteGraph(4, 3, "int");
      } break;

      case 7: {
        input_data_ = CreateTreeGraph(7, 1, "int");
      } break;

      case 8: {
        input_data_ = CreateDisconnectedGraph(5, 1, "int");
      } break;

      case 9: {
        input_data_ = CreateRandomGraph(15, 0.4, 5, "int");
      } break;

      case 10: {
        input_data_ = CreateVaryingWeightGraph(6, "int");
      } break;

      case 11: {
        input_data_ = CreateCycleGraph(6, 2, "int");
      } break;

      case 12: {
        input_data_ = CreateStarGraph(8, 3, "int");
      } break;

      case 13: {
        input_data_ = CreateGridGraph(3, 3, 1, "int");
      } break;

      case 14: {
        if (is_mpi_test_) {
          input_data_ = CreateLargeGraph(50, 1, "int");
        } else {
          input_data_ = CreateLargeGraph(30, 1, "int");
        }
      } break;

      default: {
        input_data_ = CreateSimpleGraph(3, 1, "int");
      } break;
    }

    // Для SEQ тестов вычисляем ожидаемый результат
    if (!is_mpi_test_) {
      BaranovADijkstraCrsSEQ seq_task(input_data_);
      if (seq_task.Validation() && seq_task.PreProcessing()) {
        seq_task.Run();
        expected_output_ = seq_task.GetOutput();
      }
    }
  }

 private:
  GraphCRS CreateSimpleGraph(int vertices, double base_weight, const std::string &type) {
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

    if (type == "int") {
      graph.weights = static_cast<int>(base_weight);
    } else if (type == "float") {
      graph.weights = static_cast<float>(base_weight);
    } else {
      graph.weights = base_weight;
    }

    return graph;
  }

  GraphCRS CreateLargeGraph(int vertices, double weight, const std::string &type) {
    GraphCRS graph;
    graph.vertices = vertices;
    graph.source = 0;

    graph.row_ptr.push_back(0);
    for (int i = 0; i < vertices; ++i) {
      for (int offset = 1; offset <= 3 && i + offset < vertices; ++offset) {
        graph.col_idx.push_back(i + offset);
      }
      graph.row_ptr.push_back(graph.col_idx.size());
    }

    if (type == "int") {
      graph.weights = static_cast<int>(weight);
    } else if (type == "float") {
      graph.weights = static_cast<float>(weight);
    } else {
      graph.weights = weight;
    }

    return graph;
  }

  GraphCRS CreateSingleVertexGraph() {
    GraphCRS graph;
    graph.vertices = 1;
    graph.source = 0;
    graph.row_ptr = {0, 0};
    graph.weights = 0;
    return graph;
  }

  GraphCRS CreateCompleteGraph(int vertices, double weight, const std::string &type) {
    GraphCRS graph;
    graph.vertices = vertices;
    graph.source = 0;

    graph.row_ptr.push_back(0);
    for (int i = 0; i < vertices; ++i) {
      for (int j = 0; j < vertices; ++j) {
        if (i != j) {
          graph.col_idx.push_back(j);
        }
      }
      graph.row_ptr.push_back(graph.col_idx.size());
    }

    if (type == "int") {
      graph.weights = static_cast<int>(weight);
    } else if (type == "float") {
      graph.weights = static_cast<float>(weight);
    } else {
      graph.weights = weight;
    }

    return graph;
  }

  GraphCRS CreateTreeGraph(int vertices, double weight, const std::string &type) {
    GraphCRS graph;
    graph.vertices = vertices;
    graph.source = 0;

    graph.row_ptr.push_back(0);
    for (int i = 0; i < vertices; ++i) {
      int left = 2 * i + 1;
      int right = 2 * i + 2;

      if (left < vertices) {
        graph.col_idx.push_back(left);
      }
      if (right < vertices) {
        graph.col_idx.push_back(right);
      }
      graph.row_ptr.push_back(graph.col_idx.size());
    }

    if (type == "int") {
      graph.weights = static_cast<int>(weight);
    } else if (type == "float") {
      graph.weights = static_cast<float>(weight);
    } else {
      graph.weights = weight;
    }

    return graph;
  }

  GraphCRS CreateDisconnectedGraph(int vertices, double weight, const std::string &type) {
    GraphCRS graph;
    graph.vertices = vertices;
    graph.source = 0;

    graph.row_ptr.push_back(0);
    for (int i = 0; i < vertices; ++i) {
      // Вершина 0 связана только с вершиной 1
      if (i == 0 && vertices > 1) {
        graph.col_idx.push_back(1);
      }
      graph.row_ptr.push_back(graph.col_idx.size());
    }

    if (type == "int") {
      graph.weights = static_cast<int>(weight);
    } else if (type == "float") {
      graph.weights = static_cast<float>(weight);
    } else {
      graph.weights = weight;
    }

    return graph;
  }

  GraphCRS CreateRandomGraph(int vertices, double density, double max_weight, const std::string &type) {
    GraphCRS graph;
    graph.vertices = vertices;
    graph.source = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);

    graph.row_ptr.push_back(0);
    for (int i = 0; i < vertices; ++i) {
      for (int j = 0; j < vertices; ++j) {
        if (i != j && prob_dist(gen) < density) {
          graph.col_idx.push_back(j);
        }
      }
      graph.row_ptr.push_back(graph.col_idx.size());
    }

    if (type == "int") {
      graph.weights = static_cast<int>(max_weight);
    } else if (type == "float") {
      graph.weights = static_cast<float>(max_weight);
    } else {
      graph.weights = max_weight;
    }

    return graph;
  }

  GraphCRS CreateVaryingWeightGraph(int vertices, const std::string &type) {
    return CreateSimpleGraph(vertices, 1.0, type);
  }

  GraphCRS CreateCycleGraph(int vertices, double weight, const std::string &type) {
    GraphCRS graph;
    graph.vertices = vertices;
    graph.source = 0;

    graph.row_ptr.push_back(0);
    for (int i = 0; i < vertices; ++i) {
      int next = (i + 1) % vertices;
      graph.col_idx.push_back(next);
      graph.row_ptr.push_back(graph.col_idx.size());
    }

    if (type == "int") {
      graph.weights = static_cast<int>(weight);
    } else if (type == "float") {
      graph.weights = static_cast<float>(weight);
    } else {
      graph.weights = weight;
    }

    return graph;
  }

  GraphCRS CreateStarGraph(int vertices, double weight, const std::string &type) {
    GraphCRS graph;
    graph.vertices = vertices;
    graph.source = 0;

    graph.row_ptr.push_back(0);
    for (int i = 0; i < vertices; ++i) {
      if (i == 0) {
        for (int j = 1; j < vertices; ++j) {
          graph.col_idx.push_back(j);
        }
      } else {
        graph.col_idx.push_back(0);
      }
      graph.row_ptr.push_back(graph.col_idx.size());
    }

    if (type == "int") {
      graph.weights = static_cast<int>(weight);
    } else if (type == "float") {
      graph.weights = static_cast<float>(weight);
    } else {
      graph.weights = weight;
    }

    return graph;
  }

  GraphCRS CreateGridGraph(int rows, int cols, double weight, const std::string &type) {
    int vertices = rows * cols;
    GraphCRS graph;
    graph.vertices = vertices;
    graph.source = 0;

    auto index = [cols](int r, int c) { return r * cols + c; };

    graph.row_ptr.push_back(0);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if (r > 0) {
          graph.col_idx.push_back(index(r - 1, c));
        }
        if (r < rows - 1) {
          graph.col_idx.push_back(index(r + 1, c));
        }
        if (c > 0) {
          graph.col_idx.push_back(index(r, c - 1));
        }
        if (c < cols - 1) {
          graph.col_idx.push_back(index(r, c + 1));
        }

        graph.row_ptr.push_back(graph.col_idx.size());
      }
    }

    if (type == "int") {
      graph.weights = static_cast<int>(weight);
    } else if (type == "float") {
      graph.weights = static_cast<float>(weight);
    } else {
      graph.weights = weight;
    }

    return graph;
  }

  template <typename T>
  bool CompareVectors(const std::vector<T> &a, const std::vector<T> &b, double epsilon) {
    if (a.size() != b.size()) {
      return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
      bool a_inf = std::isinf(a[i]);
      bool b_inf = std::isinf(b[i]);

      if (a_inf != b_inf) {
        return false;
      }
      if (a_inf) {
        if (std::signbit(a[i]) != std::signbit(b[i])) {
          return false;
        }
        continue;
      }

      bool a_nan = std::isnan(a[i]);
      bool b_nan = std::isnan(b[i]);

      if (a_nan != b_nan) {
        return false;
      }
      if (a_nan) {
        continue;
      }

      if (std::fabs(a[i] - b[i]) > epsilon) {
        return false;
      }
    }

    return true;
  }

 public:
  bool CheckTestOutputData(OutType &output_data) final {
    try {
      if (is_mpi_test_) {
        // Для MPI тестов проверяем корректность структуры
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized) {
          return true;
        }

        if (std::holds_alternative<std::vector<int>>(output_data)) {
          auto output = std::get<std::vector<int>>(output_data);
          const auto &graph = std::get<GraphCRS>(input_data_);
          return !output.empty() && output.size() == static_cast<size_t>(graph.vertices) && output[graph.source] == 0;
        } else if (std::holds_alternative<std::vector<float>>(output_data)) {
          auto output = std::get<std::vector<float>>(output_data);
          const auto &graph = std::get<GraphCRS>(input_data_);
          return !output.empty() && output.size() == static_cast<size_t>(graph.vertices) &&
                 std::fabs(output[graph.source]) < 1e-6;
        } else if (std::holds_alternative<std::vector<double>>(output_data)) {
          auto output = std::get<std::vector<double>>(output_data);
          const auto &graph = std::get<GraphCRS>(input_data_);
          return !output.empty() && output.size() == static_cast<size_t>(graph.vertices) &&
                 std::fabs(output[graph.source]) < 1e-9;
        }

        return false;
      } else {
        // Для SEQ тестов сравниваем с ожидаемым результатом
        if (std::holds_alternative<std::vector<int>>(output_data) &&
            std::holds_alternative<std::vector<int>>(expected_output_)) {
          return CompareVectors(std::get<std::vector<int>>(output_data), std::get<std::vector<int>>(expected_output_),
                                0.0);
        } else if (std::holds_alternative<std::vector<float>>(output_data) &&
                   std::holds_alternative<std::vector<float>>(expected_output_)) {
          return CompareVectors(std::get<std::vector<float>>(output_data),
                                std::get<std::vector<float>>(expected_output_), 1e-5);
        } else if (std::holds_alternative<std::vector<double>>(output_data) &&
                   std::holds_alternative<std::vector<double>>(expected_output_)) {
          return CompareVectors(std::get<std::vector<double>>(output_data),
                                std::get<std::vector<double>>(expected_output_), 1e-9);
        }

        return false;
      }

    } catch (...) {
      return false;
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
  bool is_mpi_test_ = false;
};

namespace {

TEST_P(BaranovADijkstraCrsFuncTests, DijkstraCRSTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {
    std::make_tuple(1, "small_int_graph"),    std::make_tuple(2, "medium_int_graph"),
    std::make_tuple(3, "float_weight_graph"), std::make_tuple(4, "double_weight_graph"),
    std::make_tuple(5, "single_vertex"),      std::make_tuple(6, "complete_graph"),
    std::make_tuple(7, "tree_graph"),         std::make_tuple(8, "disconnected_graph"),
    std::make_tuple(9, "random_graph"),       std::make_tuple(10, "varying_weights"),
    std::make_tuple(11, "cycle_graph"),       std::make_tuple(12, "star_graph"),
    std::make_tuple(13, "grid_graph"),        std::make_tuple(14, "large_graph"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BaranovADijkstraCrsMPI, InType>(kTestParam, PPC_SETTINGS_baranov_a_dijkstra_crs),
    ppc::util::AddFuncTask<BaranovADijkstraCrsSEQ, InType>(kTestParam, PPC_SETTINGS_baranov_a_dijkstra_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaranovADijkstraCrsFuncTests::PrintFuncTestName<BaranovADijkstraCrsFuncTests>;

INSTANTIATE_TEST_SUITE_P(DijkstraCRSFuncTests, BaranovADijkstraCrsFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baranov_a_dijkstra_crs
