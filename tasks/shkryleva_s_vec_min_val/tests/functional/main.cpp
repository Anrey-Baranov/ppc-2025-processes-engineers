#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cstddef>
#include <string>
#include <tuple>

#include "shkryleva_s_vec_min_val/common/include/common.hpp"
#include "shkryleva_s_vec_min_val/mpi/include/ops_mpi.hpp"
#include "shkryleva_s_vec_min_val/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shkryleva_s_vec_min_val {

class ShkrylevaRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static auto PrintTestParam(const TestType &test_param) -> std::string {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {1};
        expected_output_ = 1;
        break;
      case 1:
        input_data_ = {11, 11, 11, 11, 11, 11};
        expected_output_ = 11;
        break;
      case 2:
        input_data_ = {-981, 981};
        expected_output_ = -981;
        break;
      case 3:
        input_data_ = {-12, -67, -953, -82};
        expected_output_ = -953;
        break;
      case 4:
        input_data_ = {10000, -20000, 600000, -600000, 5050000};
        expected_output_ = -600000;
        break;
      case 5:
        input_data_ = {};
        expected_output_ = INT_MAX;
        break;
      default:
        input_data_ = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
        expected_output_ = 0;
        break;
    }
  }

  auto CheckTestOutputData(OutType &output_data) -> bool final {
    return (expected_output_ == output_data);
  }

  auto GetTestInputData() -> InType final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_{};
};

namespace {

TEST_P(ShkrylevaRunFuncTestsProcesses, FindMinValue) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(0, "single_element"), std::make_tuple(1, "all_equal"),
                                            std::make_tuple(2, "small_vector"),   std::make_tuple(3, "negative_vector"),
                                            std::make_tuple(4, "large_values"),   std::make_tuple(5, "empty_vector")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ShkrylevaSVecMinValMPI, InType>(kTestParam, PPC_SETTINGS_shkryleva_s_vec_min_val),
    ppc::util::AddFuncTask<ShkrylevaSVecMinValSEQ, InType>(kTestParam, PPC_SETTINGS_shkryleva_s_vec_min_val));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShkrylevaRunFuncTestsProcesses::PrintFuncTestName<ShkrylevaRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(VectorMinTests, ShkrylevaRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shkryleva_s_vec_min_val
