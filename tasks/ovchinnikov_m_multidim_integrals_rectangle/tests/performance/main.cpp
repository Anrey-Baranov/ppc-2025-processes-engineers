#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "ovchinnikov_m_multidim_integrals_rectangle/common/include/common.hpp"
#include "ovchinnikov_m_multidim_integrals_rectangle/mpi/include/ops_mpi.hpp"
#include "ovchinnikov_m_multidim_integrals_rectangle/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ovchinnikov_m_multidim_integrals_rectangle {

class OvchinnikovMRunPerfTestsMDIR : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 300;
  InType input_data_;

  void SetUp() override {
    int dim = 3;
    std::vector<double> lower_bounds(dim, 0.0);
    std::vector<double> upper_bounds(dim, 1.0);

    input_data_ = std::make_tuple(kCount_, dim, lower_bounds, upper_bounds);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data > 0.0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(OvchinnikovMRunPerfTestsMDIR, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, OvchinnikovMMultiDimIntegralsRectangleMPI,
                                                       OvchinnikovMMultiDimIntegralsRectangleSEQ>(
    PPC_SETTINGS_ovchinnikov_m_multidim_integrals_rectangle);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OvchinnikovMRunPerfTestsMDIR::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, OvchinnikovMRunPerfTestsMDIR, kGtestValues, kPerfTestName);

}  // namespace ovchinnikov_m_multidim_integrals_rectangle
