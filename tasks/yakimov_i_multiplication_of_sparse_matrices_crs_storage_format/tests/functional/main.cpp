#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yakimov_i_multiplication_of_sparse_matrices_crs_storage_format/common/include/common.hpp"
#include "yakimov_i_multiplication_of_sparse_matrices_crs_storage_format/mpi/include/ops_mpi.hpp"
#include "yakimov_i_multiplication_of_sparse_matrices_crs_storage_format/seq/include/ops_seq.hpp"

namespace yakimov_i_multiplication_of_sparse_matrices_crs_storage_format {

class YakimovIMultiplicationOfSparseMatricesFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    if (!std::isfinite(output_data)) {
      return false;
    }
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    InType test_id = std::get<0>(params);
    static const std::unordered_map<InType, std::pair<double, double>> kExpectedResults = {
        // значения предподсчитаны в сторонней программе для функциональных тестов
        {1, {723.404, 0.1}},     // small_square
        {2, {0.0, 1e-9}},        // rectangular_2x4
        {3, {3728.58, 0.1}},     // rectangular_4x2
        {4, {-5100.66, 0.1}},    // medium_square
        {5, {-7394.2, 0.1}},     // rectangular_3x5
        {31, {0.0, 1e-9}},       // edge_1x1
        {32, {-42992.0, 10.0}},  // edge_1x100
        {33, {-9885.96, 0.1}},   // edge_100x1
        {34, {704.299, 0.01}},   // edge_small
        {35, {0.0, 1e-9}},       // edge_all_zeros
        {36, {-3857.9, 0.1}}     // edge_high_density
    };

    auto it = kExpectedResults.find(test_id);
    if (it == kExpectedResults.end()) {
      return true;
    }

    const double expected = it->second.first;
    const double tolerance = it->second.second;

    return std::abs(output_data - expected) <= tolerance;
  }

  InType GetTestInputData() final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return std::get<0>(params);
  }
};

namespace {

TEST_P(YakimovIMultiplicationOfSparseMatricesFuncTests, SparseMatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kAllTestParam = {
    std::make_tuple(1, "small_square"),      std::make_tuple(2, "rectangular_2x4"),
    std::make_tuple(3, "rectangular_4x2"),   std::make_tuple(4, "medium_square"),
    std::make_tuple(5, "rectangular_3x5"),   std::make_tuple(31, "edge_1x1"),
    std::make_tuple(32, "edge_1x100"),       std::make_tuple(33, "edge_100x1"),
    std::make_tuple(34, "edge_small"),       std::make_tuple(35, "edge_all_zeros"),
    std::make_tuple(36, "edge_high_density")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<YakimovIMultiplicationOfSparseMatricesMPI, InType>(
                       kAllTestParam, PPC_SETTINGS_yakimov_i_multiplication_of_sparse_matrices_crs_storage_format),
                   ppc::util::AddFuncTask<YakimovIMultiplicationOfSparseMatricesSEQ, InType>(
                       kAllTestParam, PPC_SETTINGS_yakimov_i_multiplication_of_sparse_matrices_crs_storage_format));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    YakimovIMultiplicationOfSparseMatricesFuncTests::PrintFuncTestName<YakimovIMultiplicationOfSparseMatricesFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, YakimovIMultiplicationOfSparseMatricesFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace yakimov_i_multiplication_of_sparse_matrices_crs_storage_format
