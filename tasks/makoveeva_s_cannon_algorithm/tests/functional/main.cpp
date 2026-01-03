#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "makoveeva_s_cannon_algorithm/common/include/common.hpp"
#include "makoveeva_s_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "makoveeva_s_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace makoveeva_s_cannon_algorithm {

namespace {

std::vector<double> GenMatrix(int n, double seed) {
  const auto n_sz = static_cast<std::size_t>(n);
  std::vector<double> m(n_sz * n_sz);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const auto idx = (static_cast<std::size_t>(i) * n_sz) + static_cast<std::size_t>(j);
      m[idx] = seed + (0.01 * static_cast<double>(i)) - (0.02 * static_cast<double>(j)) +
               (0.001 * static_cast<double>(i * j));
    }
  }

  return m;
}

std::vector<double> MultiplyRef(const std::vector<double> &a, const std::vector<double> &b, int n) {
  const auto n_sz = static_cast<std::size_t>(n);
  std::vector<double> c(n_sz * n_sz, 0.0);

  for (int i = 0; i < n; ++i) {
    const auto i_sz = static_cast<std::size_t>(i);
    for (int k = 0; k < n; ++k) {
      const auto k_sz = static_cast<std::size_t>(k);
      const double a_ik = a[(i_sz * n_sz) + k_sz];
      for (int j = 0; j < n; ++j) {
        const auto j_sz = static_cast<std::size_t>(j);
        c[(i_sz * n_sz) + j_sz] += a_ik * b[(k_sz * n_sz) + j_sz];
      }
    }
  }

  return c;
}

bool AlmostEqualVec(const std::vector<double> &x, const std::vector<double> &y, double eps = 1e-9) {
  if (x.size() != y.size()) {
    return false;
  }

  for (std::size_t idx = 0; idx < x.size(); ++idx) {
    const double diff = std::fabs(x[idx] - y[idx]);
    const double norm = std::max({1.0, std::fabs(x[idx]), std::fabs(y[idx])});
    if (diff > eps * norm) {
      return false;
    }
  }

  return true;
}

}  // namespace

class MakoveevaSCannonAlgorithmFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<2>(GetParam());
    n_ = std::get<0>(params);

    const auto a = GenMatrix(n_, 1.0);
    const auto b = GenMatrix(n_, 2.0);

    input_ = std::make_tuple(a, b, n_);
    ref_ = MultiplyRef(a, b, n_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return AlmostEqualVec(output_data, ref_);
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  int n_ = 0;
  InType input_;
  OutType ref_;
};

TEST_P(MakoveevaSCannonAlgorithmFuncTests, CannonMatmul) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {
    std::make_tuple(1, "n1"),   std::make_tuple(2, "n2"),   std::make_tuple(4, "n4"),
    std::make_tuple(8, "n8"),   std::make_tuple(16, "n16"), std::make_tuple(20, "n20"),
    std::make_tuple(32, "n32"), std::make_tuple(3, "n3"),   std::make_tuple(7, "n7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MakoveevaSCannonAlgorithmMPI, InType>(kTestParam, PPC_SETTINGS_makoveeva_s_cannon_algorithm),
    ppc::util::AddFuncTask<MakoveevaSCannonAlgorithmSEQ, InType>(kTestParam,
                                                                 PPC_SETTINGS_makoveeva_s_cannon_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = MakoveevaSCannonAlgorithmFuncTests::PrintFuncTestName<MakoveevaSCannonAlgorithmFuncTests>;

INSTANTIATE_TEST_SUITE_P(BasicTests, MakoveevaSCannonAlgorithmFuncTests, kGtestValues, kFuncTestName);

}  // namespace makoveeva_s_cannon_algorithm
