#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kolotukhin_a_merge_sort_doubles/common/include/common.hpp"
#include "kolotukhin_a_merge_sort_doubles/mpi/include/ops_mpi.hpp"
#include "kolotukhin_a_merge_sort_doubles/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kolotukhin_a_merge_sort_doubles {

class KolotukhinAMergeSortDoublesFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "_type_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(std::get<0>(params));
    expected_output_ = std::get<1>(std::get<0>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::get<1>(output_data) == 0) {
      if (std::get<0>(output_data).size() != expected_output_.size()) {
        return false;
      }
      for (std::size_t i = 0; i < std::get<0>(output_data).size(); ++i) {
        if (std::abs(std::get<0>(output_data)[i] - expected_output_[i]) > 1e-12) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<double> expected_output_;
};

namespace {

TEST_P(KolotukhinAMergeSortDoublesFuncTests, DoubleSort) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(std::make_tuple(std::vector<double>{3.5, -2.1, 0.0, 1.1, -3.3, 2.2, -1.4, 5.6},
                                    std::vector<double>{-3.3, -2.1, -1.4, 0.0, 1.1, 2.2, 3.5, 5.6}),
                    "common_test"),

    std::make_tuple(std::make_tuple(std::vector<double>{-5.07, -3.3, -1.12, 0.4, 1.111, 3.0, 5.25},
                                    std::vector<double>{-5.07, -3.3, -1.12, 0.4, 1.111, 3.0, 5.25}),
                    "sorted_array"),

    std::make_tuple(std::make_tuple(std::vector<double>{5.0, 3.4, 1.5, 0.0, -1.0, -3.01, -5.0},
                                    std::vector<double>{-5.0, -3.01, -1.0, 0.0, 1.5, 3.4, 5.0}),
                    "reversed_array"),

    std::make_tuple(
        std::make_tuple(std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0}, std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0}),
        "all_elements_are_equal"),

    std::make_tuple(std::make_tuple(std::vector<double>{}, std::vector<double>{}), "empty_array"),

    std::make_tuple(std::make_tuple(std::vector<double>{42.0}, std::vector<double>{42.0}), "single_element")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<kolotukhin_a_merge_sort_doubles::KolotukhinAMergeSortDoublesMPI, InType>(
                       kTestParam, PPC_SETTINGS_kolotukhin_a_merge_sort_doubles),
                   ppc::util::AddFuncTask<kolotukhin_a_merge_sort_doubles::KolotukhinAMergeSortDoublesSEQ, InType>(
                       kTestParam, PPC_SETTINGS_kolotukhin_a_merge_sort_doubles));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KolotukhinAMergeSortDoublesFuncTests::PrintFuncTestName<KolotukhinAMergeSortDoublesFuncTests>;

INSTANTIATE_TEST_SUITE_P(DoubleSortTests, KolotukhinAMergeSortDoublesFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kolotukhin_a_merge_sort_doubles
