#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zaharov_g_seidel_int_met/common/include/common.hpp"
#include "zaharov_g_seidel_int_met/mpi/include/ops_mpi.hpp"
#include "zaharov_g_seidel_int_met/seq/include/ops_seq.hpp"

namespace zaharov_g_seidel_int_met {

class ZaharovGSeidelIntMetFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int system_size = std::get<0>(params);
    std::string test_name = std::get<1>(params);

    double epsilon = 1e-6;
    if (test_name == "high_precision") {
      epsilon = 1e-8;
    } else if (test_name == "low_precision") {
      epsilon = 1e-4;
    }

    input_data_ = {static_cast<double>(system_size), epsilon, 1000.0};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    int system_size = static_cast<int>(input_data_[0]);
    double epsilon = input_data_[1];

    if (output_data.size() != static_cast<std::size_t>(system_size)) {
      return false;
    }

    std::vector<std::vector<double>> A(system_size, std::vector<double>(system_size));
    std::vector<double> b(system_size);

    for (int i = 0; i < system_size; ++i) {
      b[i] = i + 1.0;
      for (int j = 0; j < system_size; ++j) {
        if (i == j) {
          A[i][j] = system_size + 1.0;
        } else {
          A[i][j] = 1.0 / (std::abs(i - j) + 1.0);
        }
      }
    }

    double residual_norm = 0.0;
    for (int i = 0; i < system_size; ++i) {
      double sum = 0.0;
      for (int j = 0; j < system_size; ++j) {
        sum += A[i][j] * output_data[j];
      }
      residual_norm += std::abs(sum - b[i]);
    }

    residual_norm /= system_size;

    return residual_norm < epsilon * 1000.0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(ZaharovGSeidelIntMetFuncTests, SeidelMethodTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(10, "small_system"),  std::make_tuple(50, "medium_system"),
    std::make_tuple(100, "large_system"), std::make_tuple(10, "high_precision"),
    std::make_tuple(20, "low_precision"), std::make_tuple(30, "medium_precision")};

const auto kSeidelMethodList = std::tuple_cat(
    ppc::util::AddFuncTask<ZaharovGSeidelIntMetMPI, InType>(kTestParam, PPC_SETTINGS_zaharov_g_seidel_int_met),
    ppc::util::AddFuncTask<ZaharovGSeidelIntMetSEQ, InType>(kTestParam, PPC_SETTINGS_zaharov_g_seidel_int_met));

const auto kGtestValues = ppc::util::ExpandToValues(kSeidelMethodList);

const auto kPerfTestName = ZaharovGSeidelIntMetFuncTests::PrintFuncTestName<ZaharovGSeidelIntMetFuncTests>;

INSTANTIATE_TEST_SUITE_P(ZaharovGSeidelIntMetTests, ZaharovGSeidelIntMetFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zaharov_g_seidel_int_met
