#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "shkryleva_s_seidel_method/common/include/common.hpp"
#include "shkryleva_s_seidel_method/mpi/include/ops_mpi.hpp"
#include "shkryleva_s_seidel_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shkryleva_s_seidel_method {

class ShkrylevaSSeidelMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(ShkrylevaSSeidelMethodFuncTests, GaussSeidelMethodTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(1, "size_1_test"),   std::make_tuple(2, "size_2_test"),   std::make_tuple(3, "size_3_test"),
    std::make_tuple(5, "size_5_test"),   std::make_tuple(10, "size_10_test"), std::make_tuple(15, "size_15_test"),
    std::make_tuple(20, "size_20_test"), std::make_tuple(25, "size_25_test"), std::make_tuple(30, "size_30_test"),
    std::make_tuple(35, "size_35_test")

};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ShkrylevaSSeidelMethodMPI, InType>(kTestParam, PPC_SETTINGS_shkryleva_s_seidel_method),
    ppc::util::AddFuncTask<ShkrylevaSSeidelMethodSEQ, InType>(kTestParam, PPC_SETTINGS_shkryleva_s_seidel_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShkrylevaSSeidelMethodFuncTests::PrintFuncTestName<ShkrylevaSSeidelMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(SeidelMethodTests, ShkrylevaSSeidelMethodFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shkryleva_s_seidel_method
