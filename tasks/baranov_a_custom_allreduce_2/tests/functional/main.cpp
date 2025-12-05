#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <variant>

#include "baranov_a_custom_allreduce_2/common/include/common.hpp"
#include "baranov_a_custom_allreduce_2/mpi/include/ops_mpi.hpp"
#include "baranov_a_custom_allreduce_2/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace baranov_a_custom_allreduce_2 {

class BaranovACustomAllreduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto param = GetParam();
    TestType test_param = std::get<2>(param);
    int test_case = std::get<0>(test_param);
    std::string task_name = std::get<1>(param);
    is_mpi_test_ = (task_name.find("mpi") != std::string::npos);

    switch (test_case) {
      case 1: {
        std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
        input_data_ = InTypeVariant{input};
        expected_input_double_ = input;
        data_type_ = MPI_DOUBLE;
      } break;
      case 2: {
        std::vector<double> input = {1.0, -2.0, 3.0, -4.0, 5.0};
        input_data_ = InTypeVariant{input};
        expected_input_double_ = input;
        data_type_ = MPI_DOUBLE;
      } break;
      case 3: {
        std::vector<double> input = {-1.0, -2.0, -3.0, -4.0, -5.0};
        input_data_ = InTypeVariant{input};
        expected_input_double_ = input;
        data_type_ = MPI_DOUBLE;
      } break;
      case 4: {
        std::vector<double> input = {1000.0, 2000.0, 3000.0};
        input_data_ = InTypeVariant{input};
        expected_input_double_ = input;
        data_type_ = MPI_DOUBLE;
      } break;
      case 5: {
        std::vector<double> input = {1.5, 2.5, 3.5};
        input_data_ = InTypeVariant{input};
        expected_input_double_ = input;
        data_type_ = MPI_DOUBLE;
      } break;
      case 6: {
        std::vector<double> input = {};
        input_data_ = InTypeVariant{input};
        expected_input_double_ = input;
        data_type_ = MPI_DOUBLE;
      } break;
      case 7: {
        std::vector<double> input = {42.0};
        input_data_ = InTypeVariant{input};
        expected_input_double_ = input;
        data_type_ = MPI_DOUBLE;
      } break;
      case 8: {
        std::vector<double> input = {0.0, 0.0, 0.0, 0.0};
        input_data_ = InTypeVariant{input};
        expected_input_double_ = input;
        data_type_ = MPI_DOUBLE;
      } break;
      case 9: {
        std::vector<int> input = {1, 2, 3, 4};
        input_data_ = InTypeVariant{input};
        expected_input_int_ = input;
        data_type_ = MPI_INT;
      } break;
      case 10: {
        std::vector<float> input = {1.1f, 2.2f, 3.3f};
        input_data_ = InTypeVariant{input};
        expected_input_float_ = input;
        data_type_ = MPI_FLOAT;
      } break;
      default: {
        std::vector<double> input = {1.0, 2.0, 3.0};
        input_data_ = InTypeVariant{input};
        expected_input_double_ = input;
        data_type_ = MPI_DOUBLE;
      } break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    try {
      if (is_mpi_test_) {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);

        if (!mpi_initialized) {
          return true;
        }

        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if (std::holds_alternative<std::vector<int>>(output_data)) {
          auto output_vec = std::get<std::vector<int>>(output_data);
          auto expected_vec = expected_input_int_;

          if (output_vec.size() != expected_vec.size()) {
            return false;
          }

          for (size_t i = 0; i < output_vec.size(); ++i) {
            int expected = expected_vec[i] * world_size;
            if (output_vec[i] != expected) {
              return false;
            }
          }
          return true;

        } else if (std::holds_alternative<std::vector<float>>(output_data)) {
          auto output_vec = std::get<std::vector<float>>(output_data);
          auto expected_vec = expected_input_float_;

          if (output_vec.size() != expected_vec.size()) {
            return false;
          }

          float epsilon = 1e-5f;
          for (size_t i = 0; i < output_vec.size(); ++i) {
            float expected = expected_vec[i] * world_size;
            if (std::abs(output_vec[i] - expected) > epsilon) {
              return false;
            }
          }
          return true;

        } else if (std::holds_alternative<std::vector<double>>(output_data)) {
          auto output_vec = std::get<std::vector<double>>(output_data);
          auto expected_vec = expected_input_double_;

          if (output_vec.size() != expected_vec.size()) {
            return false;
          }

          double epsilon = 1e-10;
          for (size_t i = 0; i < output_vec.size(); ++i) {
            double expected = expected_vec[i] * world_size;
            if (std::abs(output_vec[i] - expected) > epsilon) {
              return false;
            }
          }
          return true;
        }
        return false;
      } else {
        return output_data == input_data_;
      }
    } catch (const std::exception &) {
      return false;
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<double> expected_input_double_;
  std::vector<int> expected_input_int_;
  std::vector<float> expected_input_float_;
  MPI_Datatype data_type_;
  bool is_mpi_test_ = false;
};

namespace {

TEST_P(BaranovACustomAllreduceFuncTests, AllreduceTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(1, "positive_doubles"),   std::make_tuple(2, "mixed_doubles"),
    std::make_tuple(3, "negative_doubles"),   std::make_tuple(4, "large_doubles"),
    std::make_tuple(5, "fractional_doubles"), std::make_tuple(6, "empty_vector"),
    std::make_tuple(7, "single_element"),     std::make_tuple(8, "zeros"),
    std::make_tuple(9, "integers"),           std::make_tuple(10, "floats"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BaranovACustomAllreduceMPI, InType>(kTestParam, PPC_SETTINGS_baranov_a_custom_allreduce_2),
    ppc::util::AddFuncTask<BaranovACustomAllreduceSEQ, InType>(kTestParam, PPC_SETTINGS_baranov_a_custom_allreduce_2));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaranovACustomAllreduceFuncTests::PrintFuncTestName<BaranovACustomAllreduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(CustomAllreduceFuncTests, BaranovACustomAllreduceFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baranov_a_custom_allreduce
