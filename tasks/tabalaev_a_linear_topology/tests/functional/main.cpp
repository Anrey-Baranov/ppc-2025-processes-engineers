#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "tabalaev_a_linear_topology/common/include/common.hpp"
#include "tabalaev_a_linear_topology/mpi/include/ops_mpi.hpp"
#include "tabalaev_a_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tabalaev_a_linear_topology {

class TabalaevALinearTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "Sender_" + std::to_string(std::get<0>(test_param)) + "_Receiver_" +
           std::to_string(std::get<1>(test_param)) + "_DataSize_" + std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int sender = std::get<0>(params);
    int receiver = std::get<1>(params);
    int size = std::get<2>(params);

    std::vector<int> data(size);

    for (int i = 0; i < size; i++) {
      data[i] = (i * i) + 3;
    }

    input_data_ = std::make_tuple(sender, receiver, data);
    expected_output_ = std::vector<int>(data);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

class TabalaevALinearTopologyMpi2ProcTests : public TabalaevALinearTopologyFuncTests {
  void SetUp() override {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "Is not under mpi run\n";
      GTEST_SKIP();
    }
    TabalaevALinearTopologyFuncTests::SetUp();
  }
};

class TabalaevALinearTopologyMpi4ProcTests : public TabalaevALinearTopologyFuncTests {
  void SetUp() override {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "Is not under mpi run\n";
      GTEST_SKIP();
    }
    TabalaevALinearTopologyFuncTests::SetUp();
  }
};

class TabalaevALinearTopologyMpi6ProcTests : public TabalaevALinearTopologyFuncTests {
  void SetUp() override {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "Is not under mpi run\n";
      GTEST_SKIP();
    }
    TabalaevALinearTopologyFuncTests::SetUp();
  }
};

namespace {
TEST_P(TabalaevALinearTopologyMpi2ProcTests, Mpi2Processes) {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size < 2) {
    std::cerr << "Need at least 2 MPI processes\n";
  } else {
    ExecuteTest(GetParam());
  }
}

TEST_P(TabalaevALinearTopologyMpi4ProcTests, Mpi4Processes) {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size < 4) {
    std::cerr << "Need at least 4 MPI processes\n";
  } else {
    ExecuteTest(GetParam());
  }
}

TEST_P(TabalaevALinearTopologyMpi6ProcTests, Mpi6Processes) {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size < 6) {
    std::cerr << "Need at least 6 MPI processes\n";
  } else {
    ExecuteTest(GetParam());
  }
}

TEST_P(TabalaevALinearTopologyFuncTests, SeqTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kMpi2ProcParams = {
    std::make_tuple(0, 0, 50, "from 0 to 0"),
    std::make_tuple(0, 1, 50, "from 0 to 1"),
    std::make_tuple(1, 0, 100, "from 1 to 0"),
};

const std::array<TestType, 4> kMpi4ProcParams = {
    std::make_tuple(0, 1, 50, "from 0 to 1"),
    std::make_tuple(1, 0, 100, "from 1 to 0"),
    std::make_tuple(0, 3, 150, "from 0 to 3"),
    std::make_tuple(3, 2, 200, "from 3 to 2"),
};

const std::array<TestType, 4> kMpi6ProcParams = {
    std::make_tuple(0, 5, 250, "from 0 to 5"),
    std::make_tuple(5, 2, 250, "from 5 to 2"),
    std::make_tuple(3, 4, 250, "from 3 to 4"),
    std::make_tuple(5, 5, 250, "from 5 to 5"),
};

const std::array<TestType, 2> kSeqParams = {
    std::make_tuple(0, 0, 50, "from 0 to 0"),
    std::make_tuple(0, 1, 100, "from 0 to 1"),
};

const auto kMpiTasks2Proc = ppc::util::AddFuncTask<TabalaevALinearTopologyMPI, InType>(
    kMpi2ProcParams, PPC_SETTINGS_tabalaev_a_linear_topology);
const auto kMpiTasks4Proc = ppc::util::AddFuncTask<TabalaevALinearTopologyMPI, InType>(
    kMpi4ProcParams, PPC_SETTINGS_tabalaev_a_linear_topology);
const auto kMpiTasks6Proc = ppc::util::AddFuncTask<TabalaevALinearTopologyMPI, InType>(
    kMpi6ProcParams, PPC_SETTINGS_tabalaev_a_linear_topology);
const auto kSeqTasks =
    ppc::util::AddFuncTask<TabalaevALinearTopologySEQ, InType>(kSeqParams, PPC_SETTINGS_tabalaev_a_linear_topology);

const auto kMpiGtestValues2Proc = ppc::util::ExpandToValues(kMpiTasks2Proc);
const auto kMpiGtestValues4Proc = ppc::util::ExpandToValues(kMpiTasks4Proc);
const auto kMpiGtestValues6Proc = ppc::util::ExpandToValues(kMpiTasks6Proc);
const auto kSeqGtestValues = ppc::util::ExpandToValues(kSeqTasks);

const auto kFuncTestName = TabalaevALinearTopologyFuncTests::PrintFuncTestName<TabalaevALinearTopologyFuncTests>;

INSTANTIATE_TEST_SUITE_P(Mpi2ProcTests, TabalaevALinearTopologyMpi2ProcTests, kMpiGtestValues2Proc, kFuncTestName);

INSTANTIATE_TEST_SUITE_P(Mpi4ProcTests, TabalaevALinearTopologyMpi4ProcTests, kMpiGtestValues4Proc, kFuncTestName);

INSTANTIATE_TEST_SUITE_P(Mpi6ProcTests, TabalaevALinearTopologyMpi6ProcTests, kMpiGtestValues6Proc, kFuncTestName);

INSTANTIATE_TEST_SUITE_P(SeqTests, TabalaevALinearTopologyFuncTests, kSeqGtestValues, kFuncTestName);

}  // namespace

}  // namespace tabalaev_a_linear_topology
