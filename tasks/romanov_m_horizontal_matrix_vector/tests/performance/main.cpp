#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "romanov_m_horizontal_matrix_vector/common/include/common.hpp"
#include "romanov_m_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"
#include "romanov_m_horizontal_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace romanov_m_horizontal_matrix_vector {

class RomanovMHorizontalMatrixVectorRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const int kRows_ = 2000;
  const int kCols_ = 2000;

  InType input_data_;
  OutType expected_data_;

  void SetUp() override {
    std::vector<double> mat(static_cast<std::size_t>(kRows_) * static_cast<std::size_t>(kCols_));
    std::vector<double> vec(static_cast<std::size_t>(kCols_));

    expected_data_.resize(static_cast<std::size_t>(kRows_));

    for (int j = 0; j < kCols_; ++j) {
      vec[static_cast<std::size_t>(j)] = 0.5;
    }

    for (int i = 0; i < kRows_; ++i) {
      double sum = 0.0;
      for (int j = 0; j < kCols_; ++j) {
        const auto val = static_cast<double>((i % 10) + (j % 10));
        const std::size_t idx =
            (static_cast<std::size_t>(i) * static_cast<std::size_t>(kCols_)) + static_cast<std::size_t>(j);
        mat[idx] = val;
        sum += val * 0.5;
      }
      expected_data_[static_cast<std::size_t>(i)] = sum;
    }

    input_data_ = std::make_tuple(mat, kRows_, kCols_, vec);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_data_.size()) {
      return false;
    }
    if (std::abs(output_data.front() - expected_data_.front()) > 1e-4) {
      return false;
    }
    if (std::abs(output_data.back() - expected_data_.back()) > 1e-4) {
      return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RomanovMHorizontalMatrixVectorRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RomanovMHorizontalMatrixVectorMPI, RomanovMHorizontalMatrixVectorSEQ>(
        PPC_SETTINGS_romanov_m_horizontal_matrix_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovMHorizontalMatrixVectorRunPerfTests, kGtestValues,
                         RomanovMHorizontalMatrixVectorRunPerfTests::CustomPerfTestName);

}  // namespace romanov_m_horizontal_matrix_vector
