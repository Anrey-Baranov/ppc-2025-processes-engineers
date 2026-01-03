#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>

#include "belov_e_shell_batcher/common/include/common.hpp"
#include "belov_e_shell_batcher/mpi/include/ops_mpi.hpp"
#include "belov_e_shell_batcher/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace belov_e_shell_batcher {
class BelovEShellBatcherRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    size_t dot = test_param.find('.');
    return test_param.substr(0, dot);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_belov_e_shell_batcher, params);
    std::ifstream file(path);

    int value = 0;
    while (file >> value) {
      in_.push_back(value);
    }

    file.close();
  }
  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return in_;
  }

 private:
  InType in_;
};
namespace {
TEST_P(BelovEShellBatcherRunFuncTestsProcesses, ShellBatcherFromFiles) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {"test1.txt", "test2.txt", "test3.txt"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BelovEShellBatcherMPI, InType>(kTestParam, PPC_SETTINGS_belov_e_shell_batcher),
    ppc::util::AddFuncTask<BelovEShellBatcherSEQ, InType>(kTestParam, PPC_SETTINGS_belov_e_shell_batcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BelovEShellBatcherRunFuncTestsProcesses::PrintFuncTestName<BelovEShellBatcherRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ShellBatcher, BelovEShellBatcherRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace belov_e_shell_batcher
