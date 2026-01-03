#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "kopilov_d_ring_2/common/include/common.hpp"
#include "kopilov_d_ring_2/mpi/include/ops_mpi.hpp"
#include "kopilov_d_ring_2/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kopilov_d_ring_2 {

class KopilovDRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    auto test_name = std::get<1>(GetParam());

    // Create data
    const std::size_t vector_size = 10000000;
    input_data_.data.resize(vector_size);
    for (std::size_t i = 0; i < vector_size; ++i) {
      input_data_.data[i] = static_cast<int>(i);
    }

    // Create expected output
    expected_output_.data = input_data_.data;
    if (test_name.find("_mpi") != std::string::npos) {
      int sum_of_ranks = (world_size * (world_size - 1)) / 2;
      for (int &val : expected_output_.data) {
        val += sum_of_ranks;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.data == expected_output_.data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(KopilovDRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KopilovDRingMPI, KopilovDRingSEQ>(PPC_SETTINGS_kopilov_d_ring_2);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KopilovDRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModePerfTests, KopilovDRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kopilov_d_ring_2
