#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "golovanov_d_bcast/common/include/common.hpp"
#include "golovanov_d_bcast/mpi/include/ops_mpi.hpp"
#include "golovanov_d_bcast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace golovanov_d_bcast {

class GolovanovDBcastPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  double count_ = 1000000;

 public:
  void SetUp() override {
    std::vector<int> v_int(0);
    std::vector<float> v_float(0);
    std::vector<double> v_double(0);
    for (int i = 0; i < count_; i++) {
      v_int.push_back(i);
      v_float.push_back(static_cast<float>(i));
      v_double.push_back(static_cast<double>(i));
    }
    int main_proc = 1;
    input_data_ = std::tuple<int, int, std::vector<int>, std::vector<float>, std::vector<double>>(
        main_proc, count_, v_int, v_float, v_double);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GolovanovDBcastPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GolovanovDBcastMPI, GolovanovDBcastSEQ>(PPC_SETTINGS_golovanov_d_bcast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GolovanovDBcastPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(BcastPerfTests, GolovanovDBcastPerfTest, kGtestValues, kPerfTestName);

}  // namespace golovanov_d_bcast
