#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "ermakov_a_ring/common/include/common.hpp"
#include "ermakov_a_ring/mpi/include/ops_mpi.hpp"
#include "ermakov_a_ring/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace ermakov_a_ring {

namespace {

const int kDataSize = 40000000;
const int kDefaultSource = 0;
const int kDefaultDest = 15;

}  // namespace

static OutType CalculateExpectedPath(const InType &input, int size) {
  std::vector<int> path;
  if (size <= 0) {
    return path;
  }

  const int src = ((input.source % size) + size) % size;
  const int dst = ((input.dest % size) + size) % size;

  const int clockwise_dist = (dst - src + size) % size;
  const int counter_dist = (src - dst + size) % size;
  bool use_clockwise = (clockwise_dist <= counter_dist);

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

class ErmakovARingPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data{0, 0, {}};

  void SetUp() override {
    input_data.source = kDefaultSource;
    input_data.dest = kDefaultDest;
    input_data.data = std::vector<int>(static_cast<size_t>(kDataSize), 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int size = 0;
    if (ppc::util::IsUnderMpirun()) {
      int initialized = 0;
      MPI_Initialized(&initialized);
      if (initialized != 0) {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
      }
    }

    if (size <= 0) {
      size = std::max(input_data.source, input_data.dest) + 1;
    }

    OutType expected = CalculateExpectedPath(input_data, size);
    return output_data == expected;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(ErmakovARingPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ErmakovATestTaskMPI, ErmakovATestTaskSEQ>(PPC_SETTINGS_ermakov_a_ring);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ErmakovARingPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ErmakovARingPerfInstantiation, ErmakovARingPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ermakov_a_ring
