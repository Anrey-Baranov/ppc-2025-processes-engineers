#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>

#include "util/include/perf_test_util.hpp"
#include "yakimov_i_multiplication_of_sparse_matrices_crs_storage_format/common/include/common.hpp"
#include "yakimov_i_multiplication_of_sparse_matrices_crs_storage_format/mpi/include/ops_mpi.hpp"
#include "yakimov_i_multiplication_of_sparse_matrices_crs_storage_format/seq/include/ops_seq.hpp"

namespace yakimov_i_multiplication_of_sparse_matrices_crs_storage_format {

class YakimovIMultiplicationOfSparseMatricesPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return std::isfinite(output_data);
  }

  InType GetTestInputData() final {
    static size_t test_index = 0;
    static constexpr std::array<InType, 4> kTestSizes = {27, 28, 29, 30};
    InType result = kTestSizes.at(test_index % kTestSizes.size());
    test_index++;
    return result;
  }
};

TEST_P(YakimovIMultiplicationOfSparseMatricesPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YakimovIMultiplicationOfSparseMatricesMPI,
                                                       YakimovIMultiplicationOfSparseMatricesSEQ>(
    PPC_SETTINGS_yakimov_i_multiplication_of_sparse_matrices_crs_storage_format);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YakimovIMultiplicationOfSparseMatricesPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YakimovIMultiplicationOfSparseMatricesPerfTests, kGtestValues, kPerfTestName);

}  // namespace yakimov_i_multiplication_of_sparse_matrices_crs_storage_format
