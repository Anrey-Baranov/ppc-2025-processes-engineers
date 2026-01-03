#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "zaharov_g_seidel_int_met/common/include/common.hpp"
#include "zaharov_g_seidel_int_met/mpi/include/ops_mpi.hpp"
#include "zaharov_g_seidel_int_met/seq/include/ops_seq.hpp"

namespace zaharov_g_seidel_int_met {

class ZaharovGSeidelIntMetPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kSystemSize_ = 2000;
  const double kEpsilon_ = 1e-6;
  const int kMaxIterations_ = 1500;

  InType input_data_{};

  void SetUp() override {
    input_data_ = {static_cast<double>(kSystemSize_), kEpsilon_, static_cast<double>(kMaxIterations_)};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    if (output_data.size() != static_cast<std::size_t>(kSystemSize_)) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZaharovGSeidelIntMetPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZaharovGSeidelIntMetMPI, ZaharovGSeidelIntMetSEQ>(
    PPC_SETTINGS_zaharov_g_seidel_int_met);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZaharovGSeidelIntMetPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZaharovGSeidelIntMetPerfTest, kGtestValues, kPerfTestName);

}  // namespace zaharov_g_seidel_int_met
