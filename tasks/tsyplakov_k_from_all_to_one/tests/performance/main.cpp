#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "tsyplakov_k_from_all_to_one/common/include/common.hpp"
#include "tsyplakov_k_from_all_to_one/mpi/include/ops_mpi.hpp"
#include "tsyplakov_k_from_all_to_one/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tsyplakov_k_from_all_to_one {

template <typename T>
class TsyplakovKRunPerfTestFromAllToOne : public ppc::util::BaseRunPerfTests<InTypeT<T>, OutTypeT<T>> {
 protected:
  static constexpr unsigned int kLocalCount = 7000000;
  InTypeT<T> input_data;

  void SetUp() override {
    std::vector<T> vec(kLocalCount);
    for (unsigned int i = 0; i < kLocalCount; ++i) {
      vec[i] = static_cast<T>(i);
    }
    input_data = std::make_tuple(vec, 0);
  }

  bool CheckTestOutputData(OutTypeT<T> &output_data) final {
    return output_data == std::get<0>(input_data);
  }

  InTypeT<T> GetTestInputData() final {
    return input_data;
  }
};

using PerfTestInt = TsyplakovKRunPerfTestFromAllToOne<int>;
using PerfTestFloat = TsyplakovKRunPerfTestFromAllToOne<float>;
using PerfTestDouble = TsyplakovKRunPerfTestFromAllToOne<double>;

TEST_P(PerfTestInt, RunPerfModes) {
  ExecuteTest(GetParam());
}
TEST_P(PerfTestFloat, RunPerfModes) {
  ExecuteTest(GetParam());
}
TEST_P(PerfTestDouble, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasksInt =
    ppc::util::MakeAllPerfTasks<InTypeT<int>, TsyplakovKFromAllToOneMPI<int>, TsyplakovKFromAllToOneSEQ>(
        PPC_SETTINGS_tsyplakov_k_from_all_to_one);

const auto kAllPerfTasksFloat = ppc::util::MakeAllPerfTasks<InTypeT<float>, TsyplakovKFromAllToOneMPI<float>>(
    PPC_SETTINGS_tsyplakov_k_from_all_to_one);

const auto kAllPerfTasksDouble = ppc::util::MakeAllPerfTasks<InTypeT<double>, TsyplakovKFromAllToOneMPI<double>>(
    PPC_SETTINGS_tsyplakov_k_from_all_to_one);

INSTANTIATE_TEST_SUITE_P(IntPerf, PerfTestInt, ppc::util::TupleToGTestValues(kAllPerfTasksInt),
                         PerfTestInt::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(FloatPerf, PerfTestFloat, ppc::util::TupleToGTestValues(kAllPerfTasksFloat),
                         PerfTestFloat::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(DoublePerf, PerfTestDouble, ppc::util::TupleToGTestValues(kAllPerfTasksDouble),
                         PerfTestDouble::CustomPerfTestName);

}  // namespace
}  // namespace tsyplakov_k_from_all_to_one
