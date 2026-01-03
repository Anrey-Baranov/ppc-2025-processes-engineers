#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"
#include "badanov_a_sparse_matrix_mult_double_ccs/mpi/include/ops_mpi.hpp"
#include "badanov_a_sparse_matrix_mult_double_ccs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

class BadanovASparseMatrixMultDoubleCcsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    const auto &full_param = GetParam();
    const std::string &test_name = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int matrix_size = 1000;

    if (test_name.find("small") != std::string::npos) {
      matrix_size = 500;
    } else if (test_name.find("medium") != std::string::npos) {
      matrix_size = 2000;
    } else if (test_name.find("large") != std::string::npos) {
      matrix_size = 5000;
    } else if (test_name.find("huge") != std::string::npos) {
      matrix_size = 10000;
    }

    int rows = matrix_size;
    int inner_dim = matrix_size;
    int cols = matrix_size;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> value_dist(0.0, 10.0);
    std::bernoulli_distribution sparse_dist(0.1);

    std::vector<double> values_a;
    std::vector<int> row_indices_a;
    std::vector<int> col_pointers_a(inner_dim + 1, 0);

    int nnz_a = 0;
    for (int col = 0; col < inner_dim; ++col) {
      col_pointers_a[col] = nnz_a;
      for (int row = 0; row < rows; ++row) {
        if (sparse_dist(gen)) {
          values_a.push_back(value_dist(gen));
          row_indices_a.push_back(row);
          nnz_a++;
        }
      }
    }
    col_pointers_a[inner_dim] = nnz_a;

    std::vector<double> value_b;
    std::vector<int> row_indices_b;
    std::vector<int> col_pointers_b(cols + 1, 0);

    int nnz_b = 0;
    for (int col = 0; col < cols; ++col) {
      col_pointers_b[col] = nnz_b;
      for (int row = 0; row < inner_dim; ++row) {
        if (sparse_dist(gen)) {
          value_b.push_back(value_dist(gen));
          row_indices_b.push_back(row);
          nnz_b++;
        }
      }
    }
    col_pointers_b[cols] = nnz_b;

    test_input_ = std::make_tuple(values_a, row_indices_a, col_pointers_a, value_b, row_indices_b, col_pointers_b, rows,
                                  inner_dim, cols);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const auto &in = test_input_;
    int cols = std::get<8>(in);

    const auto &col_pointers_c = std::get<2>(output_data);

    return col_pointers_c.size() == static_cast<size_t>(cols) + 1;
  }

  InType GetTestInputData() final {
    return test_input_;
  }

 private:
  InType test_input_;
};

TEST_P(BadanovASparseMatrixMultDoubleCcsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BadanovASparseMatrixMultDoubleCcsMPI, BadanovASparseMatrixMultDoubleCcsSEQ>(
        PPC_SETTINGS_badanov_a_sparse_matrix_mult_double_ccs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BadanovASparseMatrixMultDoubleCcsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BadanovASparseMatrixMultDoubleCcsPerfTests, kGtestValues, kPerfTestName);

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
