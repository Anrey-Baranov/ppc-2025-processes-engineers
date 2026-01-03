#if 0  // NOLINT(readability-avoid-unconditional-preprocessor-if)
#  include <gtest/gtest.h>
#  include <mpi.h>

#  include <algorithm>
#  include <array>
#  include <cmath>
#  include <cstddef>
#  include <string>
#  include <tuple>
#  include <vector>

#  include "Rastvorov_K_Simple_iteration_method/common/include/common.hpp"
#  include "Rastvorov_K_Simple_iteration_method/mpi/include/ops_mpi.hpp"
#  include "Rastvorov_K_Simple_iteration_method/seq/include/ops_seq.hpp"
#  include "util/include/func_test_util.hpp"

namespace rastvorov_k_simple_iteration_method {

namespace {

inline std::vector<double> ExactSolution(int n) {
  const double t = 1.0 / (3.0 * static_cast<double>(n) - 1.0);
  return std::vector<double>(static_cast<std::size_t>(n), t);
}

inline bool NearlyEqualVec(const std::vector<double> &a, const std::vector<double> &b, double eps) {
  if (a.size() != b.size()) {
    return false;
  }
  double max_diff = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  }
  return max_diff < eps;
}

}  // namespace

class RastvorovKRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  InType input_data{};
  OutType expected_output;

  void SetUp() override {
    const TestType params = std::get<TestType>(GetParam());
    const int n = std::get<0>(params);
    input_data = n;
    expected_output = ExactSolution(n);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int initialized = 0;
    MPI_Initialized(&initialized);

    int rank = 0;
    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank != 0) {
      return true;
    }

    return NearlyEqualVec(output_data, expected_output, 1e-6);
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(RastvorovKRunFuncTestsProcesses, SimpleIterationSLAE) {
  ExecuteTest(GetParam());
}

namespace {

const std::array<TestType, 4> kTestParam = {
    std::make_tuple(1, "n1"),
    std::make_tuple(2, "n2"),
    std::make_tuple(5, "n5"),
    std::make_tuple(10, "n10"),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RastvorovKSimpleIterationMethodMPI, InType>(
                                               kTestParam, PPC_SETTINGS_Rastvorov_K_Simple_iteration_method),
                                           ppc::util::AddFuncTask<RastvorovKSimpleIterationMethodSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_Rastvorov_K_Simple_iteration_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = RastvorovKRunFuncTestsProcesses::PrintFuncTestName<RastvorovKRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SimpleIterTests, RastvorovKRunFuncTestsProcesses, kGtestValues, kTestName);

}  // namespace

}  // namespace rastvorov_k_simple_iteration_method

#endif
