#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <string>
#include <tuple>

#include "trofimov_n_linear_topology/common/include/common.hpp"
#include "trofimov_n_linear_topology/mpi/include/ops_mpi.hpp"
#include "trofimov_n_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace trofimov_n_linear_topology {

namespace {

std::string InputDataToString(const InType &in) {
  return "src" + std::to_string(in.source) + "_tgt" + std::to_string(in.target) + "_val" + std::to_string(in.value);
}

std::string MakeGTestNameSafe(const std::string &s) {
  std::string result = s;
  for (auto &c : result) {
    if (std::isalnum(c) == 0) {
      c = '_';
    }
  }
  return result;
}

}  // namespace

class TrofimovNLinearTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string name = std::get<1>(test_param);
    return MakeGTestNameSafe(name + "_" + InputDataToString(std::get<0>(test_param)));
  }

 protected:
  void SetUp() override {
    input_data_ = {.source = 0, .target = 1, .value = 42};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_.value == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(TrofimovNLinearTopologyFuncTests, LinearTransmission) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(InType{0, 0, 10}, "test1"),        std::make_tuple(InType{0, 1, 20}, "test2"),
    std::make_tuple(InType{1, 0, 30}, "test3"),        std::make_tuple(InType{0, 2, 40}, "test4"),
    std::make_tuple(InType{2, 1, 50}, "test5"),        std::make_tuple(InType{-1, 0, 10}, "invalid_src"),
    std::make_tuple(InType{0, -1, 10}, "invalid_tgt"), std::make_tuple(InType{0, 1, -10}, "invalid_val"),
    std::make_tuple(InType{0, 100, 10}, "tgt_oob"),    std::make_tuple(InType{100, 0, 10}, "src_oob"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TrofimovNLinearTopologyMPI, InType>(kTestParam, PPC_SETTINGS_trofimov_n_linear_topology),
    ppc::util::AddFuncTask<TrofimovNLinearTopologySEQ, InType>(kTestParam, PPC_SETTINGS_trofimov_n_linear_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TrofimovNLinearTopologyFuncTests::PrintFuncTestName<TrofimovNLinearTopologyFuncTests>;

INSTANTIATE_TEST_SUITE_P(LinearTopologyTests, TrofimovNLinearTopologyFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace trofimov_n_linear_topology
