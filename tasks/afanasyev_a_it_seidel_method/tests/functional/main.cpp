#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "afanasyev_a_it_seidel_method/common/include/common.hpp"
#include "afanasyev_a_it_seidel_method/mpi/include/ops_mpi.hpp"
#include "afanasyev_a_it_seidel_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace afanasyev_a_it_seidel_method {

class AfanasyevAItSeidelMethodFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, int> {
 public:
  static std::string PrintTestParam(const int &test_param) {
    return "n_" + std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    const int system_size = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_.clear();
    input_data_.push_back(static_cast<double>(system_size));
    input_data_.push_back(1e-6);                       // epsilon
    input_data_.push_back(static_cast<double>(1000));  // max iterations
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int world_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

      if (world_rank != 0) {
        return true;
      }
    }

    const int system_size = static_cast<int>(input_data_[0]);
    return output_data.size() == static_cast<std::size_t>(system_size);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(AfanasyevAItSeidelMethodFuncTests, SolveSystem) {
  ExecuteTest(GetParam());
}

namespace {

const std::array<int, 3> kTestParam = {3, 5, 10};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<AfanasyevAItSeidelMethodMPI, InType>(kTestParam, PPC_SETTINGS_afanasyev_a_it_seidel_method),
    ppc::util::AddFuncTask<AfanasyevAItSeidelMethodSEQ, InType>(kTestParam, PPC_SETTINGS_afanasyev_a_it_seidel_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = AfanasyevAItSeidelMethodFuncTests::PrintFuncTestName<AfanasyevAItSeidelMethodFuncTests>;

INSTANTIATE_TEST_SUITE_P(SolveTests, AfanasyevAItSeidelMethodFuncTests, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace afanasyev_a_it_seidel_method
