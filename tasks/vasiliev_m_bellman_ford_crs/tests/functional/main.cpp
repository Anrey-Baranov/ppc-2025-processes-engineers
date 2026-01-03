#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"
#include "vasiliev_m_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_bellman_ford_crs/seq/include/ops_seq.hpp"

namespace vasiliev_m_bellman_ford_crs {

class VasilievMBellmanFordCrsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string name = std::get<1>(test_param);
    for (auto &c : name) {
      if (!static_cast<bool>(std::isalnum(c))) {
        c = '_';
      }
    }
    return name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string filename = std::get<1>(params);
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_bellman_ford_crs, filename);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Wrong path.");
    }

    int num_vertices = 0;
    int num_edges = 0;
    file >> num_vertices >> num_edges;

    std::vector<std::tuple<int, int, int>> edges(num_edges);
    for (int i = 0; i < num_edges; i++) {
      file >> std::get<0>(edges[i]) >> std::get<1>(edges[i]) >> std::get<2>(edges[i]);
    }

    int source = 0;
    file >> source;

    expected_data_.resize(num_vertices);
    for (int i = 0; i < num_vertices; i++) {
      file >> expected_data_[i];
    }

    input_data_.row_ptr.assign(num_vertices + 1, 0);
    for (const auto &e : edges) {
      input_data_.row_ptr[std::get<0>(e) + 1]++;
    }

    for (int i = 1; i <= num_vertices; i++) {
      input_data_.row_ptr[i] += input_data_.row_ptr[i - 1];
    }

    input_data_.col_ind.resize(num_edges);
    input_data_.vals.resize(num_edges);

    std::vector<int> counter = input_data_.row_ptr;
    for (const auto &e : edges) {
      int u = std::get<0>(e);
      int idx = counter[u]++;
      input_data_.col_ind[idx] = std::get<1>(e);
      input_data_.vals[idx] = std::get<2>(e);
    }

    input_data_.source = source;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_data_;
};

namespace {

TEST_P(VasilievMBellmanFordCrsFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {TestType{0, "tree_func_test0.txt"}, TestType{1, "tree_func_test1.txt"},
                                            TestType{2, "tree_func_test2.txt"}, TestType{3, "tree_func_test3.txt"}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VasilievMBellmanFordCrsMPI, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_bellman_ford_crs),
    ppc::util::AddFuncTask<VasilievMBellmanFordCrsSEQ, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_bellman_ford_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VasilievMBellmanFordCrsFuncTests::PrintFuncTestName<VasilievMBellmanFordCrsFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VasilievMBellmanFordCrsFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace vasiliev_m_bellman_ford_crs
