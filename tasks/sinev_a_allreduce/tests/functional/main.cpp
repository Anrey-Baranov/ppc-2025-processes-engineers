#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"
#include "sinev_a_allreduce/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace sinev_a_allreduce {

class SinevAAllreduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "size_" + std::to_string(std::get<0>(test_param)) + "_type_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto param = GetParam();
    TestType test_param = std::get<2>(param);
    int vector_size = std::get<0>(test_param);
    std::string data_type = std::get<1>(test_param);
    std::string task_name = std::get<1>(param);

    is_mpi_test_ = (task_name.find("mpi") != std::string::npos);

    int rank = 0;
    if (is_mpi_test_) {
      int mpi_init = 0;
      MPI_Initialized(&mpi_init);
      if (mpi_init != 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      }
    }

    if (data_type == "int") {
      std::vector<int> vec(vector_size);
      for (int i = 0; i < vector_size; i++) {
        vec[i] = ((rank + 1) * 100) + i;
      }
      input_data_ = vec;
    } else if (data_type == "float") {
      std::vector<float> vec(vector_size);
      for (int i = 0; i < vector_size; i++) {
        auto rank_float = static_cast<float>(rank + 1);
        auto i_float = static_cast<float>(i);
        vec[i] = (rank_float * 100.0F) + i_float;
      }
      input_data_ = vec;
    } else if (data_type == "double") {
      std::vector<double> vec(vector_size);
      for (int i = 0; i < vector_size; i++) {
        auto rank_double = static_cast<double>(rank + 1);
        auto i_double = static_cast<double>(i);
        vec[i] = (rank_double * 100.0) + i_double;
      }
      input_data_ = vec;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    try {
      if (is_mpi_test_) {
        int world_size = 1;
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        if (mpi_init != 0) {
          MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        }

        int total_sum = 0;
        for (int i = 0; i < world_size; i++) {
          total_sum += (i + 1);
        }

        if (std::holds_alternative<std::vector<int>>(output_data)) {
          auto &vec = std::get<std::vector<int>>(output_data);
          auto &input_vec = std::get<std::vector<int>>(input_data_);
          if (vec.size() != input_vec.size()) {
            return false;
          }

          for (size_t i = 0; i < vec.size(); i++) {
            int expected = (total_sum * 100) + (static_cast<int>(i) * world_size);
            if (vec[i] != expected) {
              return false;
            }
          }
          return true;
        }

        if (std::holds_alternative<std::vector<float>>(output_data)) {
          auto &vec = std::get<std::vector<float>>(output_data);
          auto &input_vec = std::get<std::vector<float>>(input_data_);
          if (vec.size() != input_vec.size()) {
            return false;
          }

          for (size_t i = 0; i < vec.size(); i++) {
            auto total_sum_float = static_cast<float>(total_sum);
            auto i_float = static_cast<float>(i);
            auto world_size_float = static_cast<float>(world_size);
            float expected = (total_sum_float * 100.0F) + (i_float * world_size_float);

            if (std::fabs(vec[i] - expected) > 1e-6F) {
              return false;
            }
          }
          return true;
        }

        if (std::holds_alternative<std::vector<double>>(output_data)) {
          auto &vec = std::get<std::vector<double>>(output_data);
          auto &input_vec = std::get<std::vector<double>>(input_data_);
          if (vec.size() != input_vec.size()) {
            return false;
          }

          for (size_t i = 0; i < vec.size(); i++) {
            auto total_sum_double = static_cast<double>(total_sum);
            auto i_double = static_cast<double>(i);
            auto world_size_double = static_cast<double>(world_size);
            double expected = (total_sum_double * 100.0) + (i_double * world_size_double);
            if (std::fabs(vec[i] - expected) > 1e-9) {
              return false;
            }
          }
          return true;
        }

      } else {
        return output_data == input_data_;
      }

      return false;
    } catch (...) {
      return false;
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  bool is_mpi_test_ = false;
};

namespace {

TEST_P(SinevAAllreduceFuncTests, VectorAllreduceTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {
    std::make_tuple(1, "int"),     std::make_tuple(10, "int"),   std::make_tuple(100, "int"),
    std::make_tuple(1000, "int"),  std::make_tuple(1, "float"),  std::make_tuple(10, "float"),
    std::make_tuple(100, "float"), std::make_tuple(1, "double"), std::make_tuple(10, "double"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SinevAAllreduce, InType>(kTestParam, PPC_SETTINGS_sinev_a_allreduce),
                   ppc::util::AddFuncTask<SinevAAllreduceSEQ, InType>(kTestParam, PPC_SETTINGS_sinev_a_allreduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = SinevAAllreduceFuncTests::PrintFuncTestName<SinevAAllreduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorAllreduceTests, SinevAAllreduceFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sinev_a_allreduce
