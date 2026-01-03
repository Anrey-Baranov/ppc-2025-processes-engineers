#include <gtest/gtest.h>

#include <cstddef>

#include "util/include/perf_test_util.hpp"
#include "zyazeva_s_graham_scheme/common/include/common.hpp"
#include "zyazeva_s_graham_scheme/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_graham_scheme/seq/include/ops_seq.hpp"

namespace zyazeva_s_graham_scheme {

class ZyazevaSGrahamSchemeLargePerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data;

  void SetUp() override {
    constexpr int kNumPoints = 50000000;
    input_data.resize(kNumPoints);

    for (int i = 0; i < kNumPoints; ++i) {
      input_data[i].x = i % 5000;
      input_data[i].y = (i * 3) % 5000;  // 5
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {  // NOLINT
    if (output_data.size() < 3) {
      return true;
    }

    auto cross = [](const Point &origin, const Point &a, const Point &b) {
      return ((a.x - origin.x) * (b.y - origin.y)) - ((a.y - origin.y) * (b.x - origin.x));
    };

    const size_t n = output_data.size();
    for (size_t i = 0; i < n; ++i) {
      const Point &a = output_data[i];
      const Point &b = output_data[(i + 1) % n];
      const Point &c = output_data[(i + 2) % n];
      if (cross(a, b, c) < 0) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(ZyazevaSGrahamSchemeLargePerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZyazevaSGrahamSchemeMPI, ZyazevaSGrahamSchemeSEQ>(
    PPC_SETTINGS_zyazeva_s_graham_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZyazevaSGrahamSchemeLargePerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZyazevaSGrahamSchemeLargePerfTest, kGtestValues, kPerfTestName);

}  // namespace zyazeva_s_graham_scheme
