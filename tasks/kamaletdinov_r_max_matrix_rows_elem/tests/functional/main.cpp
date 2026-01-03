#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "kamaletdinov_r_max_matrix_rows_elem/common/include/common.hpp"
#include "kamaletdinov_r_max_matrix_rows_elem/mpi/include/ops_mpi.hpp"
#include "kamaletdinov_r_max_matrix_rows_elem/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kamaletdinov_r_max_matrix_rows_elem {

class KamaletdinovRMaxMatrixRowsElemTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    if (!std::get<1>(params).empty()) {
      GetDataFromFile(params);
    } else {
      Generate(params);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // реализована не стандратная проверка,
    // так как вектор ответа в процессе с ранком 0 имеет больший размер
    // для уменьшения времени на выделение лишней памяти
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

 private:
  InType input_data_;
  std::vector<int> correct_test_output_data_;

  void Generate(const TestType &params) {
    std::size_t m = std::get<2>(params)[0];
    std::size_t n = std::get<2>(params)[1];
    int seed = std::get<2>(params)[2];

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

  void GetDataFromFile(const TestType &params) {
    std::size_t m = 0;
    std::size_t n = 0;
    std::string local = std::get<1>(params) + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_kamaletdinov_r_max_matrix_rows_elem, local);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }
    file >> m;
    file >> n;
    std::vector<int> val(m * n);
    for (std::size_t i = 0; i < val.size(); i++) {
      file >> val[i];
    }
    input_data_ = std::make_tuple(m, n, val);
    correct_test_output_data_ = std::get<2>(params);
  }
};

namespace {

TEST_P(KamaletdinovRMaxMatrixRowsElemTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple("Matrix_3_3_from_1_to_9", "test_matrix_3_3", std::vector<int>({7, 8, 9})),
    std::make_tuple("Generate_7_7", "", std::vector<int>({7, 7, 123})),
    std::make_tuple("Generate_1000_1000", "", std::vector<int>({1000, 1000, 123})),
    std::make_tuple("Generate_77_88", "", std::vector<int>({77, 88, 123})),
    std::make_tuple("Generate_7_8", "", std::vector<int>({7, 8, 123}))};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KamaletdinovRMaxMatrixRowsElemMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kamaletdinov_r_max_matrix_rows_elem),
                                           ppc::util::AddFuncTask<KamaletdinovRMaxMatrixRowsElemSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kamaletdinov_r_max_matrix_rows_elem));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KamaletdinovRMaxMatrixRowsElemTests::PrintFuncTestName<KamaletdinovRMaxMatrixRowsElemTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KamaletdinovRMaxMatrixRowsElemTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kamaletdinov_r_max_matrix_rows_elem
