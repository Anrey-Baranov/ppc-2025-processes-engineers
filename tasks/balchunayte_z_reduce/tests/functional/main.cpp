#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "balchunayte_z_reduce/common/include/common.hpp"
#include "balchunayte_z_reduce/mpi/include/ops_mpi.hpp"
#include "balchunayte_z_reduce/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace balchunayte_z_reduce {

class BalchunayteZReduceRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int case_id = std::get<0>(params);

    input_data_ = InType{};
    input_data_.root = 0;
    expected_ = 0.0;

    switch (case_id) {
      case 0: {  // simple_positive
        input_data_.data = {1.0, 2.0, 3.0, 4.0};
        expected_ = 1.0 + 2.0 + 3.0 + 4.0;
        break;
      }
      case 1: {  // mixed_values
        input_data_.data = {-1.0, 2.5, -3.5, 4.0};
        expected_ = -1.0 + 2.5 - 3.5 + 4.0;
        break;
      }
      case 2: {  // single_element
        input_data_.data = {42.5};
        expected_ = 42.5;
        break;
      }
      case 3: {  // long_vector
        const int n = 1000;
        input_data_.data.resize(n);
        expected_ = 0.0;
        for (int i = 0; i < n; ++i) {
          input_data_.data[i] = static_cast<double>(i) * 0.5;  // 0, 0.5, 1.0, ...
          expected_ += input_data_.data[i];
        }
        break;
      }
      default: {
        input_data_.data = {1.0};
        expected_ = 1.0;
        break;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 1e-9;
    return std::fabs(output_data - expected_) < eps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_{0.0};
};

namespace {

TEST_P(BalchunayteZReduceRunFuncTestsProcesses, ReduceBasicCases) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    std::make_tuple(0, "simple_positive"),
    std::make_tuple(1, "mixed_values"),
    std::make_tuple(2, "single_element"),
    std::make_tuple(3, "long_vector"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BalchunayteZReduceMPI, InType>(kTestParam, PPC_SETTINGS_balchunayte_z_reduce),
    ppc::util::AddFuncTask<BalchunayteZReduceSEQ, InType>(kTestParam, PPC_SETTINGS_balchunayte_z_reduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    BalchunayteZReduceRunFuncTestsProcesses::PrintFuncTestName<BalchunayteZReduceRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(BalchunayteZReduceFuncTests, BalchunayteZReduceRunFuncTestsProcesses, kGtestValues,
                         kFuncTestName);

}  // namespace

}  // namespace balchunayte_z_reduce
