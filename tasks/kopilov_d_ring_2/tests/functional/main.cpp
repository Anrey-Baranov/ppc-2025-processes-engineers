#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kopilov_d_ring_2/common/include/common.hpp"
#include "kopilov_d_ring_2/mpi/include/ops_mpi.hpp"
#include "kopilov_d_ring_2/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kopilov_d_ring_2 {

class KopilovDRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    static int index = 0;
    return "VectorSize_" + std::to_string(std::get<0>(test_param).size()) + "_" + std::to_string(index++);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string test_name_str = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());

    input_data_.data = std::get<0>(params);
    expected_output_.data = std::get<0>(params);

    if (test_name_str.find("_mpi") != std::string::npos) {
      int world_size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      int sum_of_ranks = (world_size * (world_size - 1)) / 2;
      for (int &val : expected_output_.data) {
        val += sum_of_ranks;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.data == expected_output_.data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KopilovDRunFuncTestsProcesses, RingSumVector) {
  ExecuteTest(GetParam());
}

// GlobalConstantPrefix: k, Case: CamelCase
const std::array<TestType, 4> kTestParam = {
    std::make_tuple(std::vector<int>{10, 20, 30}),
    std::make_tuple(std::vector<int>{40, 50}),
    std::make_tuple(std::vector<int>{-10, 0, 10}),
    std::make_tuple(std::vector<int>(100, 5)),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KopilovDRingMPI, InType>(kTestParam, PPC_SETTINGS_kopilov_d_ring_2),
                   ppc::util::AddFuncTask<KopilovDRingSEQ, InType>(kTestParam, PPC_SETTINGS_kopilov_d_ring_2));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = KopilovDRunFuncTestsProcesses::PrintFuncTestName<KopilovDRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(RingVectorTests, KopilovDRunFuncTestsProcesses, kGtestValues, kTestName);

}  // namespace

}  // namespace kopilov_d_ring_2
