#include <gtest/gtest.h>

#include <cstddef>
#include <random>

#include "sinev_a_quicksort_with_simple_merge/common/include/common.hpp"
#include "sinev_a_quicksort_with_simple_merge/mpi/include/ops_mpi.hpp"
#include "sinev_a_quicksort_with_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sinev_a_quicksort_with_simple_merge {

class SinevAQuicksortWithSimpleMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    int size = 100000;
    input_data_.resize(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1000000);

    for (int i = 0; i < size; i++) {
      input_data_[i] = dist(gen);
    }

    // Добавляем несколько специальных значений
    input_data_[0] = -500000;
    input_data_[size - 1] = 1500000;
    input_data_[size / 2] = 0;

    // Добавляем дубликаты
    for (int i = 1; i <= 100; i++) {
      input_data_[(size / 4) + i] = 777777;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != input_data_.size()) {
      return false;
    }

    for (size_t i = 1; i < output_data.size(); i++) {
      if (output_data[i] < output_data[i - 1]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SinevAQuicksortWithSimpleMergePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SinevAQuicksortWithSimpleMergeMPI, SinevAQuicksortWithSimpleMergeSEQ>(
        PPC_SETTINGS_sinev_a_quicksort_with_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SinevAQuicksortWithSimpleMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SinevAQuicksortWithSimpleMergePerfTests, kGtestValues, kPerfTestName);

}  // namespace sinev_a_quicksort_with_simple_merge
