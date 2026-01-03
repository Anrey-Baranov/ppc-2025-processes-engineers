#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "balchunayte_z_reduce/common/include/common.hpp"
#include "balchunayte_z_reduce/mpi/include/ops_mpi.hpp"
#include "balchunayte_z_reduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace balchunayte_z_reduce {

class BalchunayteZReduceRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    constexpr int kSize = 5'000'000;

    input_data_.root = 0;
    input_data_.data.resize(kSize);

    expected_ = 0.0;
    for (int i = 0; i < kSize; ++i) {
      input_data_.data[i] = static_cast<double>(i) * 0.25;  // 0, 0.25, 0.5, ...
      expected_ += input_data_.data[i];
    }
  }

  bool CheckTestOutputData(OutType &output_data) override {
    const double abs_eps = 1e-6;
    const double rel_eps = 1e-12;

    const double diff = std::fabs(output_data - expected_);
    const double scale = std::fabs(expected_);
    const double threshold = abs_eps + (rel_eps * scale);

    return diff <= threshold;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_{0.0};
};

TEST_P(BalchunayteZReduceRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BalchunayteZReduceMPI, BalchunayteZReduceSEQ>(
    PPC_SETTINGS_balchunayte_z_reduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = BalchunayteZReduceRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BalchunayteZReduceRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace balchunayte_z_reduce
