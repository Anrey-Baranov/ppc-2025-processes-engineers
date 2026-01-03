#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

namespace {
size_t GetTypeSizeSeq(MPI_Datatype datatype) {
  if (datatype == MPI_INT) {
    return sizeof(int);
  }
  if (datatype == MPI_FLOAT) {
    return sizeof(float);
  }
  if (datatype == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 0;
}
}  // namespace

class LuchnikovETransmFrAllToOneGatherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static const size_t kDataCount = 10000000;
  MPI_Datatype data_type = MPI_INT;

  InType input_data{};

  void SetUp() override {
    const size_t type_size = sizeof(int);
    std::vector<char> data(kDataCount * type_size);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto *data_ptr = reinterpret_cast<int *>(data.data());
    for (size_t i = 0; i < kDataCount; ++i) {
      data_ptr[i] = static_cast<int>((static_cast<size_t>(rank) * kDataCount) + i);
    }

    const int root = 0;
    input_data = GatherInput{.data = data, .count = static_cast<int>(kDataCount), .datatype = data_type, .root = root};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &input = input_data;

    const auto &params = GetParam();
    const std::string task_name = std::get<1>(params);
    const bool is_mpi = task_name.find("_mpi_") != std::string::npos;

    int size = 1;
    if (is_mpi) {
      int rank = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != input.root) {
        return true;
      }
    }

    const size_t type_size = GetTypeSizeSeq(input.datatype);
    const size_t expected_size = static_cast<size_t>(input.count) * static_cast<size_t>(size) * type_size;

    return output_data.size() == expected_size;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(LuchnikovETransmFrAllToOneGatherPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnikovETransmFrAllToOneGatherMPI, LuchnikovETransmFrAllToOneGatherSEQ>(
        PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnikovETransmFrAllToOneGatherPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnikovETransmFrAllToOneGatherPerfTests, kGtestValues, kPerfTestName);

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
