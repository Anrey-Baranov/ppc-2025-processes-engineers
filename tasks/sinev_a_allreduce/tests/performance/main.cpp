#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"
#include "sinev_a_allreduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sinev_a_allreduce {

class SinevAAllreducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    auto param = GetParam();
    std::string task_name = std::get<1>(param);
    is_mpi_test_ = (task_name.find("mpi") != std::string::npos);

    const size_t vector_size = 10000000;

    int rank = 0;
    if (is_mpi_test_) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    std::vector<double> data(vector_size);

    if (is_mpi_test_) {
      for (size_t i = 0; i < vector_size; ++i) {
        auto rank_double = static_cast<double>(rank + 1);
        auto i_double = static_cast<double>(i);
        data[i] = (rank_double * 100.0) + i_double;
      }
    } else {
      for (size_t i = 0; i < vector_size; ++i) {
        data[i] = 100.0 + static_cast<double>(i);
      }
    }

    input_data_ = InTypeVariant{data};
  }

  bool CheckTestOutputData([[maybe_unused]] OutType &output_data) final {
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  bool is_mpi_test_ = false;
};

TEST_P(SinevAAllreducePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SinevAAllreduce, SinevAAllreduceSEQ>(PPC_SETTINGS_sinev_a_allreduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SinevAAllreducePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SinevAAllreducePerfTests, kGtestValues, kPerfTestName);

}  // namespace sinev_a_allreduce
