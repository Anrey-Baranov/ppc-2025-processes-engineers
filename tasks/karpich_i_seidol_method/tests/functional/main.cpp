#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "karpich_i_seidol_method/common/include/common.hpp"
#include "karpich_i_seidol_method/mpi/include/ops_mpi.hpp"
#include "karpich_i_seidol_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace karpich_i_seidol_method {

class KarpichISeidolMethodFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int n = std::get<1>(params);
    int seed = std::get<2>(params);
    task_eps_ = std::get<3>(params);

    std::vector<double> a(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    std::vector<double> b(n, 0.0);
    std::vector<double> x(n, 0.0);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> coeff(0.0, 1.0);
    std::uniform_real_distribution<double> dist_x(-10.0, 10.0);

    for (int i = 0; i < n; i++) {
      x[i] = dist_x(gen);
    }
    for (int i = 0; i < n; i++) {
      double row_sum = 0.0;
      for (int j = 0; j < n; j++) {
        if (i != j) {
          a[(i * n) + j] = coeff(gen);
          row_sum += std::abs(a[(i * n) + j]);
        }
      }
      a[(i * n) + i] = row_sum + 1.0 + coeff(gen);
    }

    for (int i = 0; i < n; i++) {
      b[i] = 0.0;
      for (int j = 0; j < n; j++) {
        b[i] += a[(i * n) + j] * x[j];
      }
    }
    input_data_ = std::make_tuple(n, a, b, task_eps_);
    check_data_ = x;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (std::size_t i = 0; i < output_data.size(); i++) {
      if (fabs((output_data[i] - check_data_[i])) > task_eps_) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<double> check_data_;
  double task_eps_ = 0.0;
};

namespace {

TEST_P(KarpichISeidolMethodFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 2> kTestParam = {std::make_tuple("genarat_with_4x_seed_666", 4, 666, 0.0001),
                                            std::make_tuple("genarate_with_14x_seed_666", 14, 666, 0.0001)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KarpichISeidolMethodMPI, InType>(kTestParam, PPC_SETTINGS_karpich_i_seidol_method),
    ppc::util::AddFuncTask<KarpichISeidolMethodSEQ, InType>(kTestParam, PPC_SETTINGS_karpich_i_seidol_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KarpichISeidolMethodFuncTestsProcesses::PrintFuncTestName<KarpichISeidolMethodFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KarpichISeidolMethodFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace karpich_i_seidol_method
