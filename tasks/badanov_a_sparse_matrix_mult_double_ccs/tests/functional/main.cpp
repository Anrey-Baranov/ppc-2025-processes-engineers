#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "badanov_a_sparse_matrix_mult_double_ccs/common/include/common.hpp"
#include "badanov_a_sparse_matrix_mult_double_ccs/mpi/include/ops_mpi.hpp"
#include "badanov_a_sparse_matrix_mult_double_ccs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace badanov_a_sparse_matrix_mult_double_ccs {

class BadanovASparseMatrixMultDoubleCcsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param)) + "x" +
           std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    const auto &full_param = GetParam();
    const std::string &task_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(full_param);
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(full_param);

    const size_t rows = std::get<0>(params);
    const size_t inner_dim = std::get<1>(params);
    const size_t cols = std::get<2>(params);

    const bool is_mpi = (task_name.find("mpi_enabled") != std::string::npos);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (is_mpi && mpi_initialized == 0) {
      GTEST_SKIP() << "MPI is not initialized (test is running without mpiexec). Skipping MPI tests.";
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> value_dist(0.0, 10.0);
    std::bernoulli_distribution sparse_dist(0.05);

    std::vector<double> values_a;
    std::vector<int> row_indices_a;
    std::vector<int> col_pointers_a(inner_dim + 1, 0);

    int nnz_a = 0;
    for (size_t col = 0; col < inner_dim; ++col) {
      col_pointers_a[col] = nnz_a;
      for (size_t row = 0; row < rows; ++row) {
        if (sparse_dist(gen)) {
          values_a.push_back(value_dist(gen));
          row_indices_a.push_back(static_cast<int>(row));
          nnz_a++;
        }
      }
    }
    col_pointers_a[inner_dim] = nnz_a;

    std::vector<double> value_b;
    std::vector<int> row_indices_b;
    std::vector<int> col_pointers_b(cols + 1, 0);

    int nnz_b = 0;
    for (size_t col = 0; col < cols; ++col) {
      col_pointers_b[col] = nnz_b;
      for (size_t row = 0; row < inner_dim; ++row) {
        if (sparse_dist(gen)) {
          value_b.push_back(value_dist(gen));
          row_indices_b.push_back(static_cast<int>(row));
          nnz_b++;
        }
      }
    }
    col_pointers_b[cols] = nnz_b;

    input_data_ = std::make_tuple(values_a, row_indices_a, col_pointers_a, value_b, row_indices_b, col_pointers_b, rows,
                                  inner_dim, cols);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &in = input_data_;
    int rows = std::get<6>(in);
    int cols = std::get<8>(in);

    const auto &value_c = std::get<0>(output_data);
    const auto &row_indices_c = std::get<1>(output_data);
    const auto &col_pointers_c = std::get<2>(output_data);

    if (col_pointers_c.size() != static_cast<size_t>(cols) + 1) {
      return false;
    }
    if (value_c.size() != row_indices_c.size()) {
      return false;
    }

    for (size_t i = 0; i < row_indices_c.size(); ++i) {
      if (row_indices_c[i] < 0 || row_indices_c[i] >= rows) {
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
};

namespace {

const std::array<TestType, 15> kTestParam = {
    std::make_tuple(10, 10, 10),    std::make_tuple(50, 50, 50),      std::make_tuple(100, 100, 100),
    std::make_tuple(200, 200, 200), std::make_tuple(100, 50, 200),    std::make_tuple(200, 100, 50),
    std::make_tuple(500, 500, 500), std::make_tuple(1000, 100, 1000), std::make_tuple(100, 1000, 100),
    std::make_tuple(300, 300, 300), std::make_tuple(400, 200, 400),   std::make_tuple(200, 400, 200),
    std::make_tuple(600, 600, 100), std::make_tuple(100, 600, 600),   std::make_tuple(800, 800, 800)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BadanovASparseMatrixMultDoubleCcsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_badanov_a_sparse_matrix_mult_double_ccs),
                                           ppc::util::AddFuncTask<BadanovASparseMatrixMultDoubleCcsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_badanov_a_sparse_matrix_mult_double_ccs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BadanovASparseMatrixMultDoubleCcsFuncTests::PrintFuncTestName<BadanovASparseMatrixMultDoubleCcsFuncTests>;

TEST_P(BadanovASparseMatrixMultDoubleCcsFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, BadanovASparseMatrixMultDoubleCcsFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace badanov_a_sparse_matrix_mult_double_ccs
