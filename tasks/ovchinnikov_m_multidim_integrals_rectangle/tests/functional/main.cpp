#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "ovchinnikov_m_multidim_integrals_rectangle/common/include/common.hpp"
#include "ovchinnikov_m_multidim_integrals_rectangle/mpi/include/ops_mpi.hpp"
#include "ovchinnikov_m_multidim_integrals_rectangle/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ovchinnikov_m_multidim_integrals_rectangle {

namespace {
double CalculateExpectedIntegral(int dim, const std::vector<double> &lower_bounds,
                                 const std::vector<double> &upper_bounds) {
  double expected = 0.0;

  for (int i = 0; i < dim; i++) {
    double integral_xi = (std::pow(upper_bounds[i], 3) - std::pow(lower_bounds[i], 3)) / 3.0;
    double product = 1.0;
    for (int j = 0; j < dim; j++) {
      if (j != i) {
        product *= (upper_bounds[j] - lower_bounds[j]);
      }
    }
    expected += integral_xi * product;
  }

  return expected;
}

double CalculateError(int n, int dim, const std::vector<double> &lower_bounds,
                      const std::vector<double> &upper_bounds) {
  double base_error = 1.0 / (n * n);
  double dim_factor = std::sqrt(static_cast<double>(dim));
  double max_range = 0.0;
  for (int i = 0; i < dim; i++) {
    max_range = std::max(max_range, upper_bounds[i] - lower_bounds[i]);
  }
  double final_error = base_error * dim_factor * max_range;
  return std::max(1e-8, std::min(final_error, 0.1));
}
}  // namespace

class OvchinnikovMRunFuncTestsMDIR : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int n = std::get<0>(test_param);
    int dim = std::get<1>(test_param);
    const auto &lower_bounds = std::get<2>(test_param);
    const auto &upper_bounds = std::get<3>(test_param);

    std::string result = "n" + std::to_string(n) + "_dim" + std::to_string(dim) + "_range";

    for (size_t i = 0; i < lower_bounds.size(); i++) {
      if (i > 0) {
        result += "_";
      }
      int lower_int = static_cast<int>(lower_bounds[i]);
      int upper_int = static_cast<int>(upper_bounds[i]);
      std::string lower_str = std::to_string(lower_int);
      std::string upper_str = std::to_string(upper_int);
      result += lower_str;
      result += "to";
      result += upper_str;
    }
    std::ranges::replace(result.begin(), result.end(), '-', 'M');

    return result;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    n_ = std::get<0>(params);
    dim_ = std::get<1>(params);
    lower_bounds_ = std::get<2>(params);
    upper_bounds_ = std::get<3>(params);

    if (n_ <= 0) {
      n_ = 10;
    }

    if (dim_ <= 0) {
      dim_ = 3;
    }

    if (lower_bounds_.size() != static_cast<size_t>(dim_)) {
      lower_bounds_ = std::vector<double>(dim_, 0.0);
    }

    if (upper_bounds_.size() != static_cast<size_t>(dim_)) {
      upper_bounds_ = std::vector<double>(dim_, 1.0);
    }

    for (int i = 0; i < dim_; i++) {
      if (lower_bounds_[i] >= upper_bounds_[i]) {
        upper_bounds_[i] = lower_bounds_[i] + 1.0;
      }
    }

    input_data_ = std::make_tuple(n_, dim_, lower_bounds_, upper_bounds_);
    expected_result_ = CalculateExpectedIntegral(dim_, lower_bounds_, upper_bounds_);
    tolerance_ = CalculateError(n_, dim_, lower_bounds_, upper_bounds_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double error = std::abs(output_data - expected_result_);
    double relative_error = error / std::max(1e-10, std::abs(expected_result_));
    return relative_error < tolerance_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  int n_ = 0;
  int dim_ = 0;
  std::vector<double> lower_bounds_;
  std::vector<double> upper_bounds_;
  double expected_result_ = 0.0;
  double tolerance_ = 0.0;
  InType input_data_;
};

namespace {

TEST_P(OvchinnikovMRunFuncTestsMDIR, MultiDimIntegralTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 27> kTestParam = {
    std::make_tuple(-10, 1, std::vector<double>{0.0}, std::vector<double>{1.0}),
    std::make_tuple(10, -2, std::vector<double>{0.0}, std::vector<double>{1.0}),
    std::make_tuple(-5, -2, std::vector<double>{0.0}, std::vector<double>{1.0}),
    std::make_tuple(0, 3, std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0}),
    std::make_tuple(10, 1, std::vector<double>{0.0}, std::vector<double>{1.0}),
    std::make_tuple(15, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(20, 3, std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0}),
    std::make_tuple(15, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{2.0, 1.0}),
    std::make_tuple(15, 2, std::vector<double>{-1.0, 0.0}, std::vector<double>{1.0, 2.0}),
    std::make_tuple(20, 3, std::vector<double>{0.0, 1.0, -1.0}, std::vector<double>{2.0, 3.0, 1.0}),
    std::make_tuple(5, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(10, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(20, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(40, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(8, 4, std::vector<double>{0.0, 0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0, 1.0}),
    std::make_tuple(6, 5, std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0}),
    std::make_tuple(5, 4, std::vector<double>{0.0, 0.0, 0.0, 0.0}, std::vector<double>{2.0, 2.0, 2.0, 2.0}),
    std::make_tuple(2, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(3, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(4, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(10, 2, std::vector<double>{-2.0, 0.5}, std::vector<double>{2.0, 3.5}),
    std::make_tuple(12, 3, std::vector<double>{-1.0, 0.0, 1.0}, std::vector<double>{1.0, 2.0, 3.0}),
    std::make_tuple(30, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(25, 3, std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0}),
    std::make_tuple(15, 2, std::vector<double>{0.5, 1.0}, std::vector<double>{2.5, 3.0}),
    std::make_tuple(18, 3, std::vector<double>{0.1, 0.2, 0.3}, std::vector<double>{1.1, 1.2, 1.3})};

// dnt touch
const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<OvchinnikovMMultiDimIntegralsRectangleSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_ovchinnikov_m_multidim_integrals_rectangle),
                                           ppc::util::AddFuncTask<OvchinnikovMMultiDimIntegralsRectangleMPI, InType>(
                                               kTestParam, PPC_SETTINGS_ovchinnikov_m_multidim_integrals_rectangle));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = OvchinnikovMRunFuncTestsMDIR::PrintFuncTestName<OvchinnikovMRunFuncTestsMDIR>;

INSTANTIATE_TEST_SUITE_P(MultiDimIntegralTests, OvchinnikovMRunFuncTestsMDIR, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ovchinnikov_m_multidim_integrals_rectangle
