#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "makoveeva_s_cannon_algorithm/common/include/common.hpp"
#include "makoveeva_s_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "makoveeva_s_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace makoveeva_s_cannon_algorithm {

namespace {

std::vector<double> GenMatrix(int n, double seed) {
  const auto n_sz = static_cast<std::size_t>(n);
  std::vector<double> m(n_sz * n_sz);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const auto idx = (static_cast<std::size_t>(i) * n_sz) + static_cast<std::size_t>(j);
      m[idx] = seed + (0.01 * static_cast<double>(i)) - (0.02 * static_cast<double>(j));
    }
  }
  return m;
}

}  // namespace

class MakoveevaSCannonAlgorithmPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kN = 729;
  InType input_data_;

  void SetUp() override {
    auto a = GenMatrix(kN, 1.0);
    auto b = GenMatrix(kN, 2.0);
    input_data_ = std::make_tuple(std::move(a), std::move(b), kN);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ВАЖНО: чтобы gtest не валился на rank != 0
    if (rank != 0) {
      return true;
    }

    const auto n_sz = static_cast<std::size_t>(kN);
    return output_data.size() == n_sz * n_sz;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MakoveevaSCannonAlgorithmPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MakoveevaSCannonAlgorithmMPI, MakoveevaSCannonAlgorithmSEQ>(
        PPC_SETTINGS_makoveeva_s_cannon_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MakoveevaSCannonAlgorithmPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MakoveevaSCannonAlgorithmPerfTests, kGtestValues, kPerfTestName);

}  // namespace makoveeva_s_cannon_algorithm
