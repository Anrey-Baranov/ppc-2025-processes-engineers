#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "ermakov_a_ring/common/include/common.hpp"
#include "ermakov_a_ring/mpi/include/ops_mpi.hpp"
#include "ermakov_a_ring/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ermakov_a_ring {

using FuncTestType = std::tuple<InType, std::string>;

static OutType CalculateExpectedPath(const InType &input, const int size) {
  std::vector<int> path;
  if (size <= 0) {
    return path;
  }

  const int src = ((input.source % size) + size) % size;
  const int dst = ((input.dest % size) + size) % size;

  const int clockwise_dist = (dst - src + size) % size;
  const int counter_dist = (src - dst + size) % size;

  bool use_clockwise = false;
  if (clockwise_dist <= counter_dist) {
    use_clockwise = true;
  }

  int cur = src;
  int steps = 0;

  if (use_clockwise) {
    while (cur != dst && steps < size) {
      path.push_back(cur);
      cur = (cur + 1) % size;
      ++steps;
    }
  } else {
    while (cur != dst && steps < size) {
      path.push_back(cur);
      cur = (cur - 1 + size) % size;
      ++steps;
    }
  }

  path.push_back(cur);
  return path;
}

class ErmakovARingFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, FuncTestType> {
 public:
  static std::string PrintTestParam(const FuncTestType &param) {
    const auto &in = std::get<0>(param);
    return "Src_" + std::to_string(in.source) + "_Dst_" + std::to_string(in.dest);
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &output) override {
    int size = 0;

    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    } else {
      size = std::max(input_.source, input_.dest) + 1;
    }

    const auto expected = CalculateExpectedPath(input_, size);
    return output == expected;
  }

 protected:
  void SetUp() override {
    const auto &param = GetParam();
    const auto &user = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(param);
    input_ = std::get<0>(user);
  }

 private:
  InType input_;
};

const std::array<FuncTestType, 13> kRingTestParams = {
    FuncTestType{InType{0, 1, {1, 2, 3}}, "0_to_1"},    FuncTestType{InType{0, 0, {10, 20}}, "0_to_0"},
    FuncTestType{InType{1, 0, {7, 8, 9}}, "1_to_0"},    FuncTestType{InType{2, 4, {4, 5, 6}}, "2_to_4"},
    FuncTestType{InType{3, 0, {11, 12, 13}}, "3_to_0"}, FuncTestType{InType{0, 3, {14, 15, 16}}, "0_to_3"},
    FuncTestType{InType{0, 2, {}}, "0_to_2"},           FuncTestType{InType{0, 10, {1, 2, 3}}, "0_to_10"},
    FuncTestType{InType{10, 1, {99}}, "10_to_1"},       FuncTestType{InType{5, 2, {7, 8}}, "5_to_2"},
    FuncTestType{InType{1, 5, {10, 20}}, "1_to_5"},     FuncTestType{InType{50, 51, {5}}, "50_to_51"},
    FuncTestType{InType{51, 50, {5}}, "51_to_50"}};

namespace {

TEST_P(ErmakovARingFuncTests, RingPath) {
  ExecuteTest(GetParam());
}

const auto kTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ErmakovATestTaskMPI, InType>(kRingTestParams, PPC_SETTINGS_ermakov_a_ring),
                   ppc::util::AddFuncTask<ErmakovATestTaskSEQ, InType>(kRingTestParams, PPC_SETTINGS_ermakov_a_ring));

const auto kGtestValues = ppc::util::ExpandToValues(kTasksList);
const auto kFuncTestName = ErmakovARingFuncTests::PrintFuncTestName<ErmakovARingFuncTests>;

INSTANTIATE_TEST_SUITE_P(RingTests, ErmakovARingFuncTests, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace ermakov_a_ring
