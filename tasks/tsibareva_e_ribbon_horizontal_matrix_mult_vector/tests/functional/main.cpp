#if 0  // NOLINT(readability-avoid-unconditional-preprocessor-if)
#  include <gtest/gtest.h>

#  include <array>
#  include <cstddef>
#  include <string>
#  include <tuple>

#  include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/common/include/common.hpp"
#  include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/mpi/include/ops_mpi.hpp"
#  include "tsibareva_e_ribbon_horizontal_matrix_mult_vector/seq/include/ops_seq.hpp"
#  include "util/include/func_test_util.hpp"
#  include "util/include/util.hpp"

namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector {

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
    input_data_ = GenerateTestData(matrix_type);
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

const std::array<TestType, 12> kTestParams = {std::make_tuple(MatrixType::kSingleConstant, "1x1_single_element"),
                                              std::make_tuple(MatrixType::kSingleRow, "1x5_single_row"),
                                              std::make_tuple(MatrixType::kSingleCol, "4x1_single_column"),
                                              std::make_tuple(MatrixType::kEmpty, "empty_matrix"),
                                              std::make_tuple(MatrixType::kSquare, "3x3_square"),
                                              std::make_tuple(MatrixType::kMoreRows, "5x2_more_rows"),
                                              std::make_tuple(MatrixType::kMoreCols, "2x5_more_cols"),
                                              std::make_tuple(MatrixType::kAllZeros, "4x4_all_zeros"),
                                              std::make_tuple(MatrixType::kPositive, "3x3_positive"),
                                              std::make_tuple(MatrixType::kNegative, "3x3_negative"),
                                              std::make_tuple(MatrixType::kMixedSigns, "2x4_mixed_signs")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TsibarevaERibbonHorizontalMatrixMultVectorMPI, InType>(
                       kTestParams, PPC_SETTINGS_tsibareva_e_ribbon_horizontal_matrix_mult_vector),
                   ppc::util::AddFuncTask<TsibarevaERibbonHorizontalMatrixMultVectorSEQ, InType>(
                       kTestParams, PPC_SETTINGS_tsibareva_e_ribbon_horizontal_matrix_mult_vector));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TsibarevaERunFuncTestsProcesses::PrintFuncTestName<TsibarevaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TsibarevaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector

#endif
