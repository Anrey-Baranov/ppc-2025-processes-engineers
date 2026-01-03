#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include "karpich_i_seidol_method/common/include/common.hpp"
#include "karpich_i_seidol_method/mpi/include/ops_mpi.hpp"
#include "karpich_i_seidol_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace karpich_i_seidol_method {

class KarpichISeidolMethodPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  std::vector<double> check_data_;
  double task_eps_ = 0.000001;
  int seed_ = 666;
  std::size_t n_ = 1000;

  void SetUp() override {
    GenerateTestData(n_, seed_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (std::size_t i = 0; i < output_data.size(); i++) {
      if (std::fabs((output_data[i] - check_data_[i])) > task_eps_) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
  void GenerateTestData(std::size_t n, int seed) {
    std::vector<double> x(n, 0.0);
    std::vector<double> a(n * n, 0.0);
    std::vector<double> b(n, 0.0);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> coeff(0.0, 1.0);
    std::uniform_real_distribution<double> dist_x(-10.0, 10.0);

    for (std::size_t i = 0; i < n; i++) {
      x[i] = dist_x(gen);
    }
    for (std::size_t i = 0; i < n; i++) {
      double row_sum = 0.0;
      for (std::size_t j = 0; j < n; j++) {
        if (i != j) {
          a[(i * n) + j] = coeff(gen);
          row_sum += std::abs(a[(i * n) + j]);
        }
      }
      a[(i * n) + i] = row_sum + 1.0 + coeff(gen);
    }

    for (std::size_t i = 0; i < n; i++) {
      b[i] = 0.0;
      for (std::size_t j = 0; j < n; j++) {
        b[i] += a[(i * n) + j] * x[j];
      }
    }
    input_data_ = std::make_tuple(n, a, b, task_eps_);
    check_data_ = x;
  }
};

TEST_P(KarpichISeidolMethodPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KarpichISeidolMethodMPI, KarpichISeidolMethodSEQ>(
    PPC_SETTINGS_karpich_i_seidol_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KarpichISeidolMethodPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KarpichISeidolMethodPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace karpich_i_seidol_method
