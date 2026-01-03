#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "timofeev_n_ribbon_scheme_only_a/common/include/common.hpp"
#include "timofeev_n_ribbon_scheme_only_a/mpi/include/ops_mpi.hpp"
#include "timofeev_n_ribbon_scheme_only_a/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace timofeev_n_ribbon_scheme_only_a {

class RibbonRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<3>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<2>(params);
    expected_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_data_.size() || output_data[0].size() != expected_data_[0].size()) {
      return false;
    }
    bool yesyes = true;
    for (size_t i = 0; i < output_data.size(); i++) {
      for (size_t j = 0; j < output_data[0].size(); j++) {
        yesyes &= output_data[i][j] == expected_data_[i][j];
      }
    }
    return yesyes;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_data_;
};

namespace {

TEST_P(RibbonRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

MatrixType v_1_2 = std::vector<std::vector<int>>(1, std::vector<int>(2, 1));
MatrixType v_2_1 = std::vector<std::vector<int>>(2, std::vector<int>(1, 1));

MatrixType v_1_3 = std::vector<std::vector<int>>(1, std::vector<int>(3, 1));
MatrixType v_3_1 = std::vector<std::vector<int>>(3, std::vector<int>(1, 1));

MatrixType v_1_4 = std::vector<std::vector<int>>(1, std::vector<int>(4, 1));
MatrixType v_4_1 = std::vector<std::vector<int>>(4, std::vector<int>(1, 1));

MatrixType v_1_5 = std::vector<std::vector<int>>(1, std::vector<int>(5, 1));
MatrixType v_5_1 = std::vector<std::vector<int>>(5, std::vector<int>(1, 1));

MatrixType const_2 = {{2}};
MatrixType const_3 = {{3}};
MatrixType const_4 = {{4}};
MatrixType const_5 = {{5}};
MatrixType const_6 = {{6}};

MatrixType o_1_1 = std::vector<std::vector<int>>(1, std::vector<int>(1, 0));

MatrixType e_2_2 = {{1, 0}, {0, 1}};
MatrixType one_2_2 = {{1, 1}, {1, 1}};
MatrixType o_2_2 = {{0, 0}, {0, 0}};
MatrixType seq_2_2 = {{1, 2}, {3, 4}};

MatrixType e_3_3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
MatrixType one_3_3 = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
MatrixType o_3_3 = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
MatrixType seq_3_3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

MatrixType e_4_4 = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
MatrixType one_4_4 = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};
MatrixType o_4_4 = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
MatrixType seq_4_4 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

MatrixType m_2_3 = {{1, 2, 3}, {4, 5, 6}};
MatrixType m_3_2 = {{1, 2}, {3, 4}, {5, 6}};
MatrixType m_2_2 = {{22, 28}, {49, 64}};

MatrixType m_3_4 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
MatrixType m_4_3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
MatrixType m_3_3 = {{70, 80, 90}, {158, 184, 210}, {246, 288, 330}};

// expected, out, in, string
const std::array<TestType, 16> kTestParam = {
    std::make_tuple(e_2_2, o_2_2, std::make_pair(e_2_2, e_2_2), "one_2_2"),
    std::make_tuple(seq_2_2, o_2_2, std::make_pair(e_2_2, seq_2_2), "seq_2_2"),
    std::make_tuple(seq_2_2, o_2_2, std::make_pair(seq_2_2, e_2_2), "seq_2_2_VV"),
    std::make_tuple(e_3_3, o_3_3, std::make_pair(e_3_3, e_3_3), "one_3_3"),
    std::make_tuple(seq_3_3, o_3_3, std::make_pair(e_3_3, seq_3_3), "seq_3_3"),
    std::make_tuple(seq_3_3, o_3_3, std::make_pair(seq_3_3, e_3_3), "seq_3_3_VV"),
    std::make_tuple(e_4_4, o_4_4, std::make_pair(e_4_4, e_4_4), "one_4_4"),
    std::make_tuple(seq_4_4, o_4_4, std::make_pair(e_4_4, seq_4_4), "seq_4_4"),
    std::make_tuple(seq_4_4, o_4_4, std::make_pair(seq_4_4, e_4_4), "seq_4_4_VV"),
    std::make_tuple(m_2_2, o_2_2, std::make_pair(m_2_3, m_3_2), "2_3_by_3_2"),
    std::make_tuple(m_3_3, o_3_3, std::make_pair(m_3_4, m_4_3), "3_4_by_4_3"),
    std::make_tuple(const_2, o_1_1, std::make_pair(v_1_2, v_2_1), "vectors_1_2_and_2_1"),
    std::make_tuple(const_3, o_1_1, std::make_pair(v_1_3, v_3_1), "vectors_1_3_and_3_1"),
    std::make_tuple(const_4, o_1_1, std::make_pair(v_1_4, v_4_1), "vectors_1_4_and_4_1"),
    std::make_tuple(const_5, o_1_1, std::make_pair(v_1_5, v_5_1), "vectors_1_5_and_5_1"),
    std::make_tuple(const_6, o_1_1, std::make_pair(const_2, const_3), "1_1_and_1_1")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TimofeevNRibbonSchemeOnlyAMPI, InType>(
                                               kTestParam, PPC_SETTINGS_timofeev_n_ribbon_scheme_only_a),
                                           ppc::util::AddFuncTask<TimofeevNRibbonSchemeOnlyASEQ, InType>(
                                               kTestParam, PPC_SETTINGS_timofeev_n_ribbon_scheme_only_a));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RibbonRunFuncTestsProcesses::PrintFuncTestName<RibbonRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, RibbonRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace timofeev_n_ribbon_scheme_only_a
