#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "shilin_n_gauss_band_horizontal_scheme/common/include/common.hpp"
#include "shilin_n_gauss_band_horizontal_scheme/mpi/include/ops_mpi.hpp"
#include "shilin_n_gauss_band_horizontal_scheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shilin_n_gauss_band_horizontal_scheme {

class ShilinNGaussBandHorizontalSchemeRunFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int matrix_size = std::get<0>(params);
    int band_width = std::get<1>(params);

    InType augmented_matrix(static_cast<size_t>(matrix_size));
    expected_output_ = std::vector<double>(static_cast<size_t>(matrix_size));
    for (int i = 0; i < matrix_size; ++i) {
      expected_output_[static_cast<size_t>(i)] = static_cast<double>(i + 1);
    }

    for (int i = 0; i < matrix_size; ++i) {
      augmented_matrix[static_cast<size_t>(i)] = std::vector<double>(static_cast<size_t>(matrix_size) + 1, 0.0);
      double sum = 0.0;

      for (int j = 0; j < matrix_size; ++j) {
        if (std::abs(i - j) <= band_width) {
          double value = (i == j) ? static_cast<double>(matrix_size + 1) : 1.0 / (std::abs(i - j) + 1.0);
          augmented_matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] = value;
          if (i != j) {
            sum += std::abs(value);
          }
        }
      }

      if (augmented_matrix[static_cast<size_t>(i)][static_cast<size_t>(i)] < sum) {
        augmented_matrix[static_cast<size_t>(i)][static_cast<size_t>(i)] = sum + 1.0;
      }
    }

    for (int i = 0; i < matrix_size; ++i) {
      double b = 0.0;
      for (int j = 0; j < matrix_size; ++j) {
        b +=
            augmented_matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] * expected_output_[static_cast<size_t>(j)];
      }
      augmented_matrix[static_cast<size_t>(i)][static_cast<size_t>(matrix_size)] = b;
    }

    input_data_ = augmented_matrix;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    const double tolerance = 1e-5;
    for (size_t i = 0; i < expected_output_.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(ShilinNGaussBandHorizontalSchemeRunFuncTestsProcesses, SolveLinearSystem) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(3, 1), std::make_tuple(4, 1), std::make_tuple(5, 2),
                                            std::make_tuple(6, 2), std::make_tuple(8, 3), std::make_tuple(10, 3)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ShilinNGaussBandHorizontalSchemeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_shilin_n_gauss_band_horizontal_scheme),
                                           ppc::util::AddFuncTask<ShilinNGaussBandHorizontalSchemeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_shilin_n_gauss_band_horizontal_scheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShilinNGaussBandHorizontalSchemeRunFuncTestsProcesses::PrintFuncTestName<
    ShilinNGaussBandHorizontalSchemeRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussEliminationTests, ShilinNGaussBandHorizontalSchemeRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace shilin_n_gauss_band_horizontal_scheme
