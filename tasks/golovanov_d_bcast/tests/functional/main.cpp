#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "golovanov_d_bcast/common/include/common.hpp"
#include "golovanov_d_bcast/mpi/include/ops_mpi.hpp"
#include "golovanov_d_bcast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace golovanov_d_bcast {

class GolovanovDBcastFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string s1 = std::to_string(std::get<0>(test_param)) + "_";
    std::string s2 = std::to_string(std::get<1>(test_param)) + "_";
    return "test" + s1 + s2;
  }

 protected:
  bool result = false;
  std::mt19937 gen{std::random_device{}()};

  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int index = std::get<0>(params);
    int n = std::get<1>(params);
    std::vector<int> v_int(0);
    std::vector<float> v_float(0);
    std::vector<double> v_double(0);
    result = std::get<2>(params);

    std::uniform_real_distribution<double> double_dist(-1000, 1000);
    std::uniform_real_distribution<float> float_dist(-1000, 1000);
    std::uniform_int_distribution<int> int_dist(-1000, 1000);

    for (int i = 0; i < n; i++) {
      v_int.push_back(int_dist(gen));
      v_float.push_back(float_dist(gen));
      v_double.push_back(double_dist(gen));
    }

    input_data_ = std::tuple<int, int, std::vector<int>, std::vector<float>, std::vector<double>>(index, n, v_int,
                                                                                                  v_float, v_double);
  }
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == result;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(GolovanovDBcastFuncTest, TestTest1) {
  ExecuteTest(GetParam());
}
const std::array<TestType, 2> kTestParam = {TestType(0, 15, true), TestType(1, 15, true)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<GolovanovDBcastMPI, InType>(kTestParam, PPC_SETTINGS_golovanov_d_bcast),
                   ppc::util::AddFuncTask<GolovanovDBcastSEQ, InType>(kTestParam, PPC_SETTINGS_golovanov_d_bcast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GolovanovDBcastFuncTest::PrintFuncTestName<GolovanovDBcastFuncTest>;

INSTANTIATE_TEST_SUITE_P(BcastFunTests, GolovanovDBcastFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace golovanov_d_bcast
