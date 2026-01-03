#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include "karpich_i_matrix_elem_sum/common/include/common.hpp"
#include "karpich_i_matrix_elem_sum/mpi/include/ops_mpi.hpp"
#include "karpich_i_matrix_elem_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace karpich_i_matrix_elem_sum {

class KarpichIMatrixElemSumPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  std::size_t n = 10000;
  std::size_t m = 10000;

 private:
  std::int64_t correct_test_output_data_ = 0;
  InType input_data_;

  void SetUp() override {
    input_data_ = std::make_tuple(n, m, GenMatrix(n, m, 777));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == correct_test_output_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  std::vector<int> GenMatrix(std::size_t rows, std::size_t cols, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> idis(0, 1000000);

    std::vector<int> res((rows * cols));
    correct_test_output_data_ = 0;

    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        const std::size_t idx = (row * cols) + col;
        res[idx] = idis(gen);
        correct_test_output_data_ += res[idx];
      }
    }
    return res;
  }
};

TEST_P(KarpichIMatrixElemSumPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KarpichIMatrixElemSumMPI, KarpichIMatrixElemSumSEQ>(
    PPC_SETTINGS_karpich_i_matrix_elem_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = KarpichIMatrixElemSumPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KarpichIMatrixElemSumPerfTest, kGtestValues, kPerfTestName);

}  // namespace karpich_i_matrix_elem_sum
