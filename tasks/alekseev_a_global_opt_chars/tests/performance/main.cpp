#include <gtest/gtest.h>

#include <functional>

#include "alekseev_a_global_opt_chars/common/include/common.hpp"
#include "alekseev_a_global_opt_chars/mpi/include/ops_mpi.hpp"
#include "alekseev_a_global_opt_chars/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace alekseev_a_global_opt_chars {

namespace {

double EllipsoidPerfFunc(double x, double y) {
  return (2.0 * x * x) + (0.5 * y * y);
}

}  // namespace

class AlekseevAGlobalOptCharsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = InType{};

    input_data_.func = EllipsoidPerfFunc;

    input_data_.x_min = -6.0;
    input_data_.x_max = 6.0;
    input_data_.y_min = -6.0;
    input_data_.y_max = 6.0;

    input_data_.epsilon = 0.0002;
    input_data_.r_param = 2.2;
    input_data_.max_iterations = 2500;
  }

  bool CheckTestOutputData(OutType &output) final {
    return output.iterations >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(AlekseevAGlobalOptCharsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, AlekseevAGlobalOptCharsMPI, AlekseevAGlobalOptCharsSEQ>(
    PPC_SETTINGS_alekseev_a_global_opt_chars);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = AlekseevAGlobalOptCharsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, AlekseevAGlobalOptCharsPerfTests, kGtestValues, kPerfTestName);

}  // namespace alekseev_a_global_opt_chars
