#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "romanov_m_horizontal_matrix_vector/common/include/common.hpp"
#include "romanov_m_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"
#include "romanov_m_horizontal_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace romanov_m_horizontal_matrix_vector {

class RomanovMHorizontalMatrixVectorRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const int rows = std::get<0>(params);
    const int cols = std::get<1>(params);

    std::vector<double> matrix(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    std::vector<double> vec(static_cast<std::size_t>(cols));
    OutType expected(static_cast<std::size_t>(rows));

    for (int j = 0; j < cols; ++j) {
      vec[static_cast<std::size_t>(j)] = 1.0;
    }

    for (int i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (int j = 0; j < cols; ++j) {
        const std::size_t idx =
            (static_cast<std::size_t>(i) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(j);
        matrix[idx] = static_cast<double>(i + j);
        sum += static_cast<double>(i + j);
      }
      expected[static_cast<std::size_t>(i)] = sum;
    }

    input_data_ = std::make_tuple(matrix, rows, cols, vec);
    expected_result_ = expected;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_result_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_result_[i]) > 1e-5) {
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
  OutType expected_result_;
};

namespace {

TEST_P(RomanovMHorizontalMatrixVectorRunFuncTests, RomanovMHorizontalMatrixVector) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {std::make_tuple(6, 6), std::make_tuple(4, 2), std::make_tuple(2, 4),
                                            std::make_tuple(1, 1), std::make_tuple(2, 1), std::make_tuple(1, 2),
                                            std::make_tuple(2, 2), std::make_tuple(3, 7)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RomanovMHorizontalMatrixVectorMPI, InType>(
                                               kTestParam, PPC_SETTINGS_romanov_m_horizontal_matrix_vector),
                                           ppc::util::AddFuncTask<RomanovMHorizontalMatrixVectorSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_romanov_m_horizontal_matrix_vector));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(
    MatrixVectorMultiplicationTests, RomanovMHorizontalMatrixVectorRunFuncTests, kGtestValues,
    RomanovMHorizontalMatrixVectorRunFuncTests::PrintFuncTestName<RomanovMHorizontalMatrixVectorRunFuncTests>);

}  // namespace

}  // namespace romanov_m_horizontal_matrix_vector
