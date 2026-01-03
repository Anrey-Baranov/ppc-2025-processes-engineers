#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "pankov_matrix_vector/common/include/common.hpp"
#include "pankov_matrix_vector/mpi/include/ops_mpi.hpp"
#include "pankov_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace pankov_matrix_vector {

using LocalTestType = std::tuple<InType, OutType>;

class PankovMatrixVectorRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, LocalTestType> {
 public:
  static std::string PrintTestParam(const LocalTestType &test_param) {
    const auto &input = std::get<0>(test_param);
    const auto &expect = std::get<1>(test_param);
    std::string result = "matrix_" + std::to_string(input.matrix.size()) + "x" +
                         (input.matrix.empty() ? "0" : std::to_string(input.matrix[0].size()));
    result += "_vector_" + std::to_string(input.vector.size());
    if (!expect.empty()) {
      result += "_result_" + std::to_string(static_cast<int>(expect[0]));
      if (expect.size() > 1) {
        result += "_" + std::to_string(static_cast<int>(expect[expect.size() - 1]));
      }
    }
    return result;
  }

 protected:
  void SetUp() override {
    LocalTestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    const double epsilon = 1e-9;
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > epsilon) {
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

const std::array<LocalTestType, 6> kTestParam = {
    LocalTestType{InType{{{1.0, 2.0}, {3.0, 4.0}}, {5.0, 6.0}}, {17.0, 39.0}},
    LocalTestType{InType{{{1.0}}, {1.0}}, {1.0}},
    LocalTestType{InType{{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, {1.0, 2.0, 3.0}}, {14.0, 32.0}},
    LocalTestType{InType{{{1.0, 0.0}, {0.0, 1.0}}, {2.0, 3.0}}, {2.0, 3.0}},
    LocalTestType{InType{{{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}}, {1.0, 2.0, 3.0}}, {2.0, 4.0, 6.0}},
    LocalTestType{InType{{{1.0, 1.0, 1.0}}, {1.0, 1.0, 1.0}}, {3.0}},
};

TEST_P(PankovMatrixVectorRunFuncTestsProcesses, MatrixVectorMultiplication) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PankovMatrixVectorMPI, InType>(kTestParam, PPC_SETTINGS_pankov_matrix_vector),
    ppc::util::AddFuncTask<PankovMatrixVectorSEQ, InType>(kTestParam, PPC_SETTINGS_pankov_matrix_vector));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kFuncTestName =
    PankovMatrixVectorRunFuncTestsProcesses::PrintFuncTestName<PankovMatrixVectorRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixVectorTests, PankovMatrixVectorRunFuncTestsProcesses, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace pankov_matrix_vector
