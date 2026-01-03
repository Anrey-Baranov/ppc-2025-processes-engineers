#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>

#include "afanasyev_a_it_seidel_method/common/include/common.hpp"
#include "afanasyev_a_it_seidel_method/mpi/include/ops_mpi.hpp"
#include "afanasyev_a_it_seidel_method/seq/include/ops_seq.hpp"
#include "performance/include/performance.hpp"
#include "task/include/task.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace afanasyev_a_it_seidel_method {

class AfanasyevAItSeidelMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kSystemSize = 10000;
  static constexpr double kEpsilon = 1e-6;
  static constexpr int kMaxIterations = 3000000;

  InType input_data;

  void SetUp() override {
    input_data.clear();
    input_data.push_back(static_cast<double>(kSystemSize));
    input_data.push_back(kEpsilon);
    input_data.push_back(static_cast<double>(kMaxIterations));
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return output_data.size() == static_cast<size_t>(kSystemSize);
  }

  InType GetTestInputData() override {
    return input_data;
  }

 public:
  static std::string CustomPerfTestName(
      const ::testing::TestParamInfo<ppc::util::PerfTestParam<InType, OutType>> &info) {
    auto base_name = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);
    auto mode = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);

    std::string mode_str = (mode == ppc::performance::PerfResults::TypeOfRunning::kPipeline) ? "pipeline" : "task";

    return mode_str + "_" + base_name;
  }
};

TEST_P(AfanasyevAItSeidelMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

template <typename TaskType, typename InputType>
auto MakePerfTaskTuples() {
  std::string name;
  if constexpr (std::is_same_v<TaskType, AfanasyevAItSeidelMethodMPI>) {
    name = "afanasyev_a_it_seidel_method_mpi";
  } else if constexpr (std::is_same_v<TaskType, AfanasyevAItSeidelMethodSEQ>) {
    name = "afanasyev_a_it_seidel_method_seq";
  } else {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist;
    name = "unknown_task_" + std::to_string(dist(gen));
  }
  auto task_lambda = [](const InputType &in) { return ppc::task::TaskGetter<TaskType, InputType>(in); };

  return std::make_tuple(std::make_tuple(task_lambda, name, ppc::performance::PerfResults::TypeOfRunning::kPipeline),
                         std::make_tuple(task_lambda, name, ppc::performance::PerfResults::TypeOfRunning::kTaskRun));
}

}  // namespace

const auto kAllPerfTasks = std::tuple_cat(MakePerfTaskTuples<AfanasyevAItSeidelMethodMPI, InType>(),
                                          MakePerfTaskTuples<AfanasyevAItSeidelMethodSEQ, InType>());

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunModeTests, AfanasyevAItSeidelMethodPerfTests, kGtestValues,
                         AfanasyevAItSeidelMethodPerfTests::CustomPerfTestName);

}  // namespace afanasyev_a_it_seidel_method
