#include <gtest/gtest.h>

#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_bubble_sort/common/include/common.hpp"
#include "vasiliev_m_bubble_sort/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_bubble_sort/seq/include/ops_seq.hpp"

namespace vasiliev_m_bubble_sort {

class VasilievMBubbleSortPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_bubble_sort, "test_vector_perf.txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Wrong path.");
    }

    input_data_.clear();
    int value = 0;

    while (file >> value) {
      input_data_.push_back(value);
    }

    std::vector<int> basic_vec = input_data_;

    for (int i = 0; i < 8; i++) {
      input_data_.insert(input_data_.end(), basic_vec.begin(), basic_vec.end());
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(VasilievMBubbleSortPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VasilievMBubbleSortMPI, VasilievMBubbleSortSEQ>(
    PPC_SETTINGS_vasiliev_m_bubble_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VasilievMBubbleSortPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VasilievMBubbleSortPerfTests, kGtestValues, kPerfTestName);

}  // namespace vasiliev_m_bubble_sort
