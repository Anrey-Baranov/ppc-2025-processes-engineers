#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "samoylenko_i_simple_iter_method/common/include/common.hpp"
#include "samoylenko_i_simple_iter_method/mpi/include/ops_mpi.hpp"
#include "samoylenko_i_simple_iter_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace samoylenko_i_simple_iter_method {

class SamoylenkoISimpleIterMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kCount = 3000;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    int rank = 0;
    int is_mpi_init = 0;
    MPI_Initialized(&is_mpi_init);
    if (is_mpi_init != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    // only rank 0 has the answer
    if (rank > 0) {
      return true;
    }

    // solution vector should have size n and contain non-trivial values
    if (output_data.size() != static_cast<size_t>(kCount)) {
      return false;
    }

    double max_diff = 0.0;
    for (int i = 0; i < kCount; ++i) {
      double ax_i = 4.0 * output_data[i];
      if (i > 0) {
        ax_i += output_data[i - 1];
      }
      if (i < kCount - 1) {
        ax_i += output_data[i + 1];
      }
      max_diff = std::max(max_diff, std::fabs(ax_i - 1.0));
    }
    return max_diff < 1e-6;
  }

  InType GetTestInputData() override {
    return input_data_;
  }
};

TEST_P(SamoylenkoISimpleIterMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SamoylenkoISimpleIterMethodMPI, SamoylenkoISimpleIterMethodSEQ>(
        PPC_SETTINGS_samoylenko_i_simple_iter_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SamoylenkoISimpleIterMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SamoylenkoISimpleIterMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace samoylenko_i_simple_iter_method
