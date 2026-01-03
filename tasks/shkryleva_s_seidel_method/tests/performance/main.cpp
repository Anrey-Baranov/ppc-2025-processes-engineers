#include <gtest/gtest.h>

#include "shkryleva_s_seidel_method/common/include/common.hpp"
#include "shkryleva_s_seidel_method/mpi/include/ops_mpi.hpp"
#include "shkryleva_s_seidel_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkryleva_s_seidel_method {

class ShkrylevaSSeidelMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const int kCount_ = 1000;
  InType input_data_{};

 protected:
  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return input_data_ == output_data;
  }

  InType GetTestInputData() override {
    return input_data_;
  }
};

TEST_P(ShkrylevaSSeidelMethodPerfTests, SeidelMethodPerformanceTest) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ShkrylevaSSeidelMethodMPI, ShkrylevaSSeidelMethodSEQ>(
    PPC_SETTINGS_shkryleva_s_seidel_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShkrylevaSSeidelMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShkrylevaSSeidelMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace shkryleva_s_seidel_method
