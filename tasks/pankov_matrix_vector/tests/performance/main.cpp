#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "pankov_matrix_vector/common/include/common.hpp"
#include "pankov_matrix_vector/mpi/include/ops_mpi.hpp"
#include "pankov_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pankov_matrix_vector {

class PankovMatrixVectorRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    const std::size_t matrix_size = 10000;
    std::vector<std::vector<double>> matrix(matrix_size);
    std::vector<double> vector(matrix_size);

    for (std::size_t i = 0; i < matrix_size; ++i) {
      matrix[i].resize(matrix_size);
      for (std::size_t j = 0; j < matrix_size; ++j) {
        matrix[i][j] = static_cast<double>(i + j);
      }
      vector[i] = static_cast<double>(i + 1);
    }

    input_data_.matrix = matrix;
    input_data_.vector = vector;

    expected_output_.resize(matrix_size);
    for (std::size_t i = 0; i < matrix_size; ++i) {
      expected_output_[i] = 0.0;
      for (std::size_t j = 0; j < matrix_size; ++j) {
        expected_output_[i] += matrix[i][j] * vector[j];
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    const double epsilon = 1e-6;
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
};

TEST_P(PankovMatrixVectorRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PankovMatrixVectorMPI, PankovMatrixVectorSEQ>(
    PPC_SETTINGS_pankov_matrix_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = PankovMatrixVectorRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PankovMatrixVectorRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace pankov_matrix_vector
