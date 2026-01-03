#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kondrashova_v_ring_topology/common/include/common.hpp"
#include "kondrashova_v_ring_topology/mpi/include/ops_mpi.hpp"
#include "kondrashova_v_ring_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kondrashova_v_ring_topology {

class KondrashovaVRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int source = std::get<0>(test_param);
    int recipient = std::get<1>(test_param);
    const std::string &name = std::get<3>(test_param);
    return "src" + std::to_string(source) + "_dst" + std::to_string(recipient) + "_" + name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int source = std::get<0>(params);
    int recipient = std::get<1>(params);
    std::vector<int> data = std::get<2>(params);

    input_data_.source = source;
    input_data_.recipient = recipient;
    input_data_.data = data;
    expected_output_ = data;

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      int world_size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      input_data_.source = source % world_size;
      input_data_.recipient = recipient % world_size;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int world_size = 0;
      int rank = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      // SEQ-режим (один процесс)
      if (world_size == 1) {
        return output_data == expected_output_;
      }

      // MPI: проверяем только на получателе
      if (rank == input_data_.recipient) {
        return output_data == expected_output_;
      }

      return true;
    }

    // Без MPI
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KondrashovaVRunFuncTestsProcesses, RingTopologyTransfer) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(0, 1, std::vector<int>{1, 2, 3, 4, 5}, "simple_transfer"),
    std::make_tuple(0, 3, std::vector<int>{10, 20, 30}, "first_to_third"),
    std::make_tuple(3, 0, std::vector<int>{100, 200}, "last_to_first"),
    std::make_tuple(0, 0, std::vector<int>{42, 43, 44}, "same_process"),
    std::make_tuple(1, 2, std::vector<int>{999}, "single_element"),
    std::make_tuple(1, 3, std::vector<int>{-1, -2, -3, 0, 1, 2, 3}, "negative_numbers")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KondrashovaVRingTopologyMPI, InType>(kTestParam, PPC_SETTINGS_kondrashova_v_ring_topology),
    ppc::util::AddFuncTask<KondrashovaVRingTopologySEQ, InType>(kTestParam, PPC_SETTINGS_kondrashova_v_ring_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KondrashovaVRunFuncTestsProcesses::PrintFuncTestName<KondrashovaVRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(RingTopologyTests, KondrashovaVRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kondrashova_v_ring_topology
