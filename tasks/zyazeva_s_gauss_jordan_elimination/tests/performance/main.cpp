#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zyazeva_s_gauss_jordan_elimination/common/include/common.hpp"
#include "zyazeva_s_gauss_jordan_elimination/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_gauss_jordan_elimination/seq/include/ops_seq.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

class ZyazevaSRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int n = 1000;

    input_data_.clear();
    input_data_.resize(n);
    solutions_.clear();

    for (int i = 0; i < n; ++i) {
      solutions_.push_back(static_cast<float>((i % 5) + 1));
    }

    for (int i = 0; i < n; ++i) {
      input_data_[i].resize(n + 1);

      for (int j = 0; j < n; ++j) {
        if (i == j) {
          input_data_[i][j] = static_cast<float>(n * 2) + static_cast<float>(i % 10);
        } else if (std::abs(i - j) <= 2) {
          input_data_[i][j] = 0.5F;
        } else {
          input_data_[i][j] = 0.01F;
        }
      }
      input_data_[i][n] = 0.0F;
      for (int j = 0; j < n; ++j) {
        input_data_[i][n] += input_data_[i][j] * solutions_[j];
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank > 0) {
        return true;
      }
    }

    if (output_data.empty()) {
      return false;
    }

    if (output_data.size() != input_data_.size()) {
      return false;
    }

    const float k_eps = 1e-2F;

    int correct_count = 0;
    int total_count = static_cast<int>(output_data.size());

    for (size_t i = 0; i < output_data.size(); ++i) {
      float expected = solutions_[i];
      float diff = std::abs(output_data[i] - expected);

      if (diff <= k_eps) {
        correct_count++;
      }
    }

    float accuracy = static_cast<float>(correct_count) / static_cast<float>(total_count);
    return accuracy >= 0.95F;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<float> solutions_;
};

TEST_P(ZyazevaSRunPerfTestProcesses, GaussJordanPerformanceTest) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZyazevaSGaussJordanElMPI, ZyazevaSGaussJordanElSEQ>(
    PPC_SETTINGS_zyazeva_s_gauss_jordan_elimination);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZyazevaSRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(GaussJordanPerfTests, ZyazevaSRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace zyazeva_s_gauss_jordan_elimination
