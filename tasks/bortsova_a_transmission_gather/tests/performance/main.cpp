#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "bortsova_a_transmission_gather/common/include/common.hpp"
#include "bortsova_a_transmission_gather/mpi/include/ops_mpi.hpp"
#include "bortsova_a_transmission_gather/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace bortsova_a_transmission_gather {

class BortsovaATransmissionGatherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 30000000;
  InType input_data_{};
  bool is_mpi_test_ = false;

  void SetUp() override {
    input_data_.send_data.resize(kDataSize);
    for (int idx = 0; idx < kDataSize; ++idx) {
      input_data_.send_data[static_cast<std::size_t>(idx)] = 1.0 + static_cast<double>(idx);
    }
    input_data_.root = 0;

    std::string test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    is_mpi_test_ = test_name.find("_mpi_") != std::string::npos;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.recv_data.empty()) {
      return false;
    }

    int world_size = 1;
    if (is_mpi_test_ && ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    std::size_t expected_size = input_data_.send_data.size() * static_cast<std::size_t>(world_size);
    if (output_data.recv_data.size() != expected_size) {
      return false;
    }

    for (std::size_t ii = 0; ii < input_data_.send_data.size(); ++ii) {
      if (output_data.recv_data[ii] != input_data_.send_data[ii]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BortsovaATransmissionGatherPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BortsovaATransmissionGatherMPI, BortsovaATransmissionGatherSEQ>(
        PPC_SETTINGS_bortsova_a_transmission_gather);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BortsovaATransmissionGatherPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BortsovaATransmissionGatherPerfTests, kGtestValues, kPerfTestName);

}  // namespace bortsova_a_transmission_gather
