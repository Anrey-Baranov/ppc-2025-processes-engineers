#include <gtest/gtest.h>

#include <cstddef>
#include <random>

#include "shkryleva_s_vec_min_val/common/include/common.hpp"
#include "shkryleva_s_vec_min_val/mpi/include/ops_mpi.hpp"
#include "shkryleva_s_vec_min_val/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkryleva_s_vec_min_val {

class ShkrylevaSVecMinValPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kVectorSize = 100000000;

 protected:
  void SetUp() override {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(-1000, 1000);

    input_data_.resize(kVectorSize);
    for (size_t i = 0; i < kVectorSize; i++) {
      input_data_[i] = dist(gen);
    }

    expected_min_ = -1500;
    input_data_[kVectorSize / 2] = expected_min_;
  }

  auto CheckTestOutputData(OutType &output_data) -> bool final {
    return expected_min_ == output_data;
  }

  auto GetTestInputData() -> InType final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_min_{};
};

TEST_P(ShkrylevaSVecMinValPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ShkrylevaSVecMinValMPI, ShkrylevaSVecMinValSEQ>(
    PPC_SETTINGS_shkryleva_s_vec_min_val);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShkrylevaSVecMinValPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShkrylevaSVecMinValPerfTests, kGtestValues, kPerfTestName);

}  // namespace shkryleva_s_vec_min_val
