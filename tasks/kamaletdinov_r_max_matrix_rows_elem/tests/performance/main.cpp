#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include "kamaletdinov_r_max_matrix_rows_elem/common/include/common.hpp"
#include "kamaletdinov_r_max_matrix_rows_elem/mpi/include/ops_mpi.hpp"
#include "kamaletdinov_r_max_matrix_rows_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kamaletdinov_r_max_matrix_rows_elem {

class KamaletdinovRMaxMatrixRowsElemPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  std::vector<int> correct_test_output_data_;
  InType input_data_;

  void SetUp() override {
    Generate(10000, 10000, 123);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (std::size_t i = 0; i < correct_test_output_data_.size(); i++) {
      if (output_data[i] != correct_test_output_data_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  void Generate(std::size_t m, std::size_t n, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> idis(-10, 20);

    std::vector<int> val(m * n);
    std::vector<int> answer(n);
    // задание начальных значений для ответа
    // первая строка матрицы задает максимальные значнечения для элементов столбцов
    for (std::size_t i = 0; i < n; i++) {
      val[i] = idis(gen);
      answer[i] = val[i];
    }
    // генерация остальной матрицы, вектора ответа
    for (std::size_t i = 1; i < m; i++) {
      for (std::size_t j = 0; j < n; j++) {
        val[(i * n) + j] = idis(gen);
        answer[j] = std::max(answer[j], val[(i * n) + j]);
      }
    }
    input_data_ = std::make_tuple(m, n, val);
    correct_test_output_data_ = answer;

    // debug output
    //  std::string deb = "\n\n-----------\n";
    //  for(std::size_t i = 0; i < m; i++) {
    //    for(std::size_t j = 0; j < n; j++) {
    //      deb += std::to_string(val[i*n + j]) + " ";
    //    }
    //    deb += "\n";
    //  }
    //  std::cout << deb;
    //  std::cout << "----------\n";
    //  for(std::size_t i = 0; i < n; i++) {
    //    std::cout << answer[i] << " ";
    //  }
    //  std::cout << std::endl;
  }
};

TEST_P(KamaletdinovRMaxMatrixRowsElemPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KamaletdinovRMaxMatrixRowsElemMPI, KamaletdinovRMaxMatrixRowsElemSEQ>(
        PPC_SETTINGS_kamaletdinov_r_max_matrix_rows_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KamaletdinovRMaxMatrixRowsElemPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KamaletdinovRMaxMatrixRowsElemPerfTest, kGtestValues, kPerfTestName);

}  // namespace kamaletdinov_r_max_matrix_rows_elem
