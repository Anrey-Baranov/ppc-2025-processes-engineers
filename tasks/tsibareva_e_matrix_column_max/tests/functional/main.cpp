#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "tsibareva_e_matrix_column_max/common/include/common.hpp"
#include "tsibareva_e_matrix_column_max/mpi/include/ops_mpi.hpp"
#include "tsibareva_e_matrix_column_max/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tsibareva_e_matrix_column_max {

class TsibarevaERunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string description = std::get<1>(test_param);
    return description;
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    MatrixType matrix_type = std::get<0>(params);
    input_data_ = GenerateMatrixFunc(matrix_type);
    expected_output_ = GenerateExpectedOutput(matrix_type);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(TsibarevaERunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 20> kTestParams = {{std::make_tuple(MatrixType::kSingleConstant, "1x1_single"),
                                               std::make_tuple(MatrixType::kSingleRow, "1x10_single_row"),
                                               std::make_tuple(MatrixType::kSingleCol, "3x1_single_col"),

                                               std::make_tuple(MatrixType::kAllZeros, "5x5_all_zeros"),
                                               std::make_tuple(MatrixType::kConstant, "5x5_constant"),

                                               std::make_tuple(MatrixType::kMaxFirst, "6x4_max_first"),
                                               std::make_tuple(MatrixType::kMaxLast, "6x4_max_last"),
                                               std::make_tuple(MatrixType::kMaxMiddle, "6x4_max_middle"),

                                               std::make_tuple(MatrixType::kAscending, "8x8_ascending_simple"),
                                               std::make_tuple(MatrixType::kDescending, "8x8_descending_simple"),
                                               std::make_tuple(MatrixType::kDiagonalDominant, "8x8_diagonal_dom"),
                                               std::make_tuple(MatrixType::kSparse, "8x8_sparse"),
                                               std::make_tuple(MatrixType::kNegative, "8x8_negative"),

                                               std::make_tuple(MatrixType::kSquareSmall, "2x2_square_small"),
                                               std::make_tuple(MatrixType::kVertical, "10x4_vertical"),
                                               std::make_tuple(MatrixType::kHorizontal, "5x10_horizontal"),
                                               std::make_tuple(MatrixType::kCheckerboard, "7x7_checkerboard"),

                                               std::make_tuple(MatrixType::kEmpty, "empty_matrix"),
                                               std::make_tuple(MatrixType::kZeroColumns, "zero_columns_matrix"),
                                               std::make_tuple(MatrixType::kNonRectangular, "non_rectangular_matrix")}};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TsibarevaEMatrixColumnMaxMPI, InType>(
                                               kTestParams, PPC_SETTINGS_tsibareva_e_matrix_column_max),
                                           ppc::util::AddFuncTask<TsibarevaEMatrixColumnMaxSEQ, InType>(
                                               kTestParams, PPC_SETTINGS_tsibareva_e_matrix_column_max));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TsibarevaERunFuncTestsProcesses::PrintFuncTestName<TsibarevaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TsibarevaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tsibareva_e_matrix_column_max
