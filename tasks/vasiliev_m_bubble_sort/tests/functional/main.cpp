#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_bubble_sort/common/include/common.hpp"
#include "vasiliev_m_bubble_sort/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_bubble_sort/seq/include/ops_seq.hpp"

namespace vasiliev_m_bubble_sort {

class VasilievMBubbleSortFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_bubble_sort, "test_vectors_func.txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Wrong path.");
    }

    test_vectors_.clear();
    std::string line;

    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }

      std::stringstream ss(line);
      std::vector<int> input;
      std::vector<int> expected;
      int value = 0;
      while (ss >> value) {
        input.push_back(value);
        ss >> std::ws;
        if (ss.peek() == ';') {
          ss.get();
          break;
        }
      }

      while (ss >> value) {
        expected.push_back(value);
      }

      test_vectors_.emplace_back(input, expected);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int index = std::get<0>(params);
    input_data_ = test_vectors_[index].first;
    expected_output_ = test_vectors_[index].second;
    return input_data_;
  }

 private:
  std::vector<std::pair<std::vector<int>, std::vector<int>>> test_vectors_;
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(VasilievMBubbleSortFuncTests, VectorSort) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(0, "case1"), std::make_tuple(1, "case2"), std::make_tuple(2, "case3"), std::make_tuple(3, "case4"),
    std::make_tuple(4, "case5"), std::make_tuple(5, "case6"), std::make_tuple(6, "case7"), std::make_tuple(7, "case8")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VasilievMBubbleSortMPI, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_bubble_sort),
    ppc::util::AddFuncTask<VasilievMBubbleSortSEQ, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_bubble_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VasilievMBubbleSortFuncTests::PrintFuncTestName<VasilievMBubbleSortFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VasilievMBubbleSortFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace vasiliev_m_bubble_sort
