#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "posternak_a_radix_merge_sort/common/include/common.hpp"
#include "posternak_a_radix_merge_sort/mpi/include/ops_mpi.hpp"
#include "posternak_a_radix_merge_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace posternak_a_radix_merge_sort {

class PosternakARadixMergeSortPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000000;

  InType input_data_;

  // Заполняем массив: чередование знаков, каждый пятый элемент - удвоенное значение позиции
  void SetUp() override {
    input_data_.clear();
    input_data_.reserve(kCount_);
    for (int i = 0; i < kCount_; i++) {
      int value = 0;
      if (i % 2 == 0) {
        value = -i;
      } else {
        value = i;
      }
      if (i % 5 == 0) {
        value *= 2;
      }
      input_data_.push_back(value);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (input_data_.size() != output_data.size()) {
      return false;
    }

    for (std::size_t i = 1; i < output_data.size(); i += 1000) {  // Проверяем каждую 1000-ю пару
      if (output_data[i] < output_data[i - 1]) {
        return false;
      }
    }

    // Проверяем первый и последний элементы
    const auto real_min = *std::ranges::min_element(input_data_);
    const auto real_max = *std::ranges::max_element(input_data_);

    return (output_data[0] == real_min) && (output_data[output_data.size() - 1] == real_max);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PosternakARadixMergeSortPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PosternakARadixMergeSortMPI, PosternakARadixMergeSortSEQ>(
        PPC_SETTINGS_posternak_a_radix_merge_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PosternakARadixMergeSortPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PosternakARadixMergeSortPerfTest, kGtestValues, kPerfTestName);

}  // namespace posternak_a_radix_merge_sort
