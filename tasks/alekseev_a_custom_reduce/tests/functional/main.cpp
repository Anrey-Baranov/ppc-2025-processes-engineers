#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "alekseev_a_custom_reduce/common/include/common.hpp"
#include "alekseev_a_custom_reduce/mpi/include/ops_mpi.hpp"
#include "alekseev_a_custom_reduce/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace alekseev_a_custom_reduce {

class AlekseevACustomReduceRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int case_id = std::get<0>(params);

    input_data_ = InType{};
    input_data_.root = 0;
    expected_ = 0.0;

    switch (case_id) {
      case 0: {
        input_data_.data = {1.0, 2.0, 3.0, 4.0, 5.0};
        expected_ = 15.0;
        break;
      }
      case 1: {
        input_data_.data = {-10.0, 20.0, -30.0, 40.0, -50.0, 60.0};
        expected_ = 30.0;
        break;
      }
      case 2: {
        input_data_.data = {42.0};
        expected_ = 42.0;
        break;
      }
      case 3: {
        input_data_.data = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
        expected_ = 550.0;
        break;
      }
      case 4: {
        input_data_.data = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        expected_ = 0.0;
        break;
      }
      case 5: {
        input_data_.data = {25.0, 25.0, 25.0, 25.0, 25.0};
        expected_ = 125.0;
        break;
      }
      case 6: {
        input_data_.data = {-15.0, -15.0, -15.0, -15.0};
        expected_ = -60.0;
        break;
      }
      case 7: {
        input_data_.data = {100.0, -100.0, 50.0, -50.0, 25.0, -25.0};
        expected_ = 0.0;
        break;
      }
      case 8: {
        input_data_.data = {0.1, 0.2, 0.3, 0.4, 0.5};
        expected_ = 1.5;
        break;
      }
      case 9: {
        input_data_.data = {-1.5, 2.5, -3.5, 4.5, -5.5};
        expected_ = -3.5;
        break;
      }
      case 10: {
        input_data_.data = {1.0, 3.0, 5.0, 7.0, 9.0};
        expected_ = 25.0;
        break;
      }
      case 11: {
        input_data_.data = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
        expected_ = 42.0;
        break;
      }
      case 12: {
        input_data_.data = {1.0, 3.0, 6.0, 10.0, 15.0};
        expected_ = 35.0;
        break;
      }
      case 13: {
        input_data_.data = {5.0, 10.0, 15.0, 20.0, 25.0, 30.0};
        expected_ = 105.0;
        break;
      }
      case 14: {
        input_data_.data = {2.0, 3.0, 5.0, 7.0, 11.0, 13.0};
        expected_ = 41.0;
        break;
      }
      case 15: {
        input_data_.data = {1.0, 2.0, 4.0, 8.0, 16.0};
        expected_ = 31.0;
        break;
      }
      case 16: {
        input_data_.data = {17.0, -23.0, 42.0, -8.0, 95.0};
        expected_ = 123.0;
        break;
      }
      case 17: {
        input_data_.data = {-45.0, 67.0, -12.0, 88.0, -33.0, 21.0};
        expected_ = 86.0;
        break;
      }
      case 18: {
        input_data_.data = {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75};
        expected_ = 7.0;
        break;
      }
      case 19: {
        input_data_.data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,
                            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
        expected_ = 210.0;
        break;
      }
      default: {
        input_data_.data = {1.0};
        expected_ = 1.0;
        break;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int initialized = 0;
    const int mpi_ok = MPI_Initialized(&initialized);

    int rank = 0;
    if ((mpi_ok == MPI_SUCCESS) && (initialized != 0)) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    const double eps = 1e-9;
    if (rank == 0) {
      return std::fabs(output_data - expected_) < eps;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_{0.0};
};

namespace {

TEST_P(AlekseevACustomReduceRunFuncTestsProcesses, ReduceBasicCases) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 20> kTestParam = {
    std::make_tuple(0, "sum_1_to_5"),          std::make_tuple(1, "mixed_integers"),
    std::make_tuple(2, "single_number"),       std::make_tuple(3, "ten_numbers_sequence"),
    std::make_tuple(4, "all_zeros"),           std::make_tuple(5, "all_same_positive"),
    std::make_tuple(6, "all_same_negative"),   std::make_tuple(7, "opposite_pairs"),
    std::make_tuple(8, "small_fractions"),     std::make_tuple(9, "mixed_fractions"),
    std::make_tuple(10, "odd_numbers"),        std::make_tuple(11, "even_numbers"),
    std::make_tuple(12, "triangular_numbers"), std::make_tuple(13, "multiples_of_5"),
    std::make_tuple(14, "prime_numbers"),      std::make_tuple(15, "powers_of_two"),
    std::make_tuple(16, "random_pattern_1"),   std::make_tuple(17, "random_pattern_2"),
    std::make_tuple(18, "more_fractions"),     std::make_tuple(19, "twenty_numbers"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<AlekseevACustomReduceMPI, InType>(kTestParam, PPC_SETTINGS_alekseev_a_custom_reduce),
    ppc::util::AddFuncTask<AlekseevACustomReduceSEQ, InType>(kTestParam, PPC_SETTINGS_alekseev_a_custom_reduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    AlekseevACustomReduceRunFuncTestsProcesses::PrintFuncTestName<AlekseevACustomReduceRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(AlekseevACustomReduceFuncTests, AlekseevACustomReduceRunFuncTestsProcesses, kGtestValues,
                         kFuncTestName);

}  // namespace

}  // namespace alekseev_a_custom_reduce
