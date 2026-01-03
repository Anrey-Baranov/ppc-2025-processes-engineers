#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "karpich_i_matrix_elem_sum/common/include/common.hpp"
#include "karpich_i_matrix_elem_sum/mpi/include/ops_mpi.hpp"
#include "karpich_i_matrix_elem_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace karpich_i_matrix_elem_sum {

class KarpichIMatrixElemSumTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::size_t n = 0;
    std::size_t m = 0;
    std::string first_param = std::get<0>(params);
    if (first_param == "gen") {
      n = std::get<1>(params);
      m = n;
      std::vector<int> val = GenMatrix(n, m, 777);
      input_data_ = std::make_tuple(n, m, val);
    } else {
      std::string local = first_param + ".txt";
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_karpich_i_matrix_elem_sum, local);
      std::ifstream file(abs_path);
      if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + abs_path);
      }

      file >> n;
      file >> m;
      std::vector<int> val(n * m);
      for (std::size_t i = 0; i < val.size(); i++) {
        file >> val[i];
      }
      input_data_ = std::make_tuple(n, m, val);
      correct_test_output_data_ = std::get<1>(params);
    }
  }
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == correct_test_output_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::int64_t correct_test_output_data_ = 0;
  std::vector<int> GenMatrix(std::size_t n, std::size_t m, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> idis(0, 1000000);
    std::vector<int> res(n * m);
    correct_test_output_data_ = 0;

    for (std::size_t i = 0; i < (n * m); i++) {
      res[i] = idis(gen);
      correct_test_output_data_ += res[i];
    }
    return res;
  }
};

namespace {

TEST_P(KarpichIMatrixElemSumTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple("test_matrix_3_3", 9),   std::make_tuple("test_matrix_5_5", 110),
    std::make_tuple("test_matrix_7_7", 219), std::make_tuple("test_matrix_10_10", 450),
    std::make_tuple("test_matrix_3_8", 106), std::make_tuple("test_matrix_4_6", 105),
    std::make_tuple("test_matrix_6_3", 81),  std::make_tuple("gen", 100)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KarpichIMatrixElemSumMPI, InType>(kTestParam, PPC_SETTINGS_karpich_i_matrix_elem_sum),
    ppc::util::AddFuncTask<KarpichIMatrixElemSumSEQ, InType>(kTestParam, PPC_SETTINGS_karpich_i_matrix_elem_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KarpichIMatrixElemSumTests::PrintFuncTestName<KarpichIMatrixElemSumTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KarpichIMatrixElemSumTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace karpich_i_matrix_elem_sum
