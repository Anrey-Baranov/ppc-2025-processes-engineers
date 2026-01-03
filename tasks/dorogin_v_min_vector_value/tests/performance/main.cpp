#include <gtest/gtest.h>

#include <cstddef>
#include <random>

#include "dorogin_v_min_vector_value/common/include/common.hpp"
#include "dorogin_v_min_vector_value/mpi/include/ops_mpi.hpp"
#include "dorogin_v_min_vector_value/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dorogin_v_min_vector_value {

class DoroginVMinVectorValueRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 50000000;

 protected:
  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000, 1000);

    input_vec_.resize(kSize);
    for (size_t i = 0; i < kSize; i++) {
      input_vec_[i] = dist(gen);
    }

    expected_res_ = -10000;
    input_vec_[kSize / 3] = expected_res_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_res_ == output_data;
  }

  InType GetTestInputData() final {
    return input_vec_;
  }

 private:
  InType input_vec_;
  OutType expected_res_{};
};

TEST_P(DoroginVMinVectorValueRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, DoroginVMinVectorValueMPI, DoroginVMinVectorValueSEQ>(
    PPC_SETTINGS_dorogin_v_min_vector_value);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DoroginVMinVectorValueRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DoroginVMinVectorValueRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace dorogin_v_min_vector_value
