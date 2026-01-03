#include "zyazeva_s_gauss_jordan_elimination/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "zyazeva_s_gauss_jordan_elimination/common/include/common.hpp"

namespace zyazeva_s_gauss_jordan_elimination {

ZyazevaSGaussJordanElSEQ::ZyazevaSGaussJordanElSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp = in;
  GetInput() = std::move(temp);
  GetOutput() = std::vector<float>();
}

bool ZyazevaSGaussJordanElSEQ::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  std::size_t n = matrix.size();

  return std::ranges::all_of(matrix, [n](const auto &row) { return row.size() == n + 1; });
}

bool ZyazevaSGaussJordanElSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<float>();
  return true;
}

namespace {

void KNormalizeCurrentRow(std::vector<std::vector<float>> &a, int i, int n) {
  float pivot = a[i][i];
  for (int k = i; k <= n; k++) {
    a[i][k] /= pivot;
  }
}

void KEliminateColumn(std::vector<std::vector<float>> &a, int i, int n) {
  for (int j = 0; j < n; j++) {
    if (j != i) {
      float factor = a[j][i];
      for (int k = i; k <= n; k++) {
        a[j][k] -= factor * a[i][k];
      }
    }
  }
}

std::vector<float> ExtractSolutions(const std::vector<std::vector<float>> &a, int n) {
  std::vector<float> solutions(n);
  for (int i = 0; i < n; i++) {
    solutions[i] = a[i][n];
  }
  return solutions;
}

}  // namespace

bool ZyazevaSGaussJordanElSEQ::RunImpl() {
  std::vector<std::vector<float>> a = GetInput();
  int n = static_cast<int>(a.size());

  for (int i = 0; i < n; i++) {
    KNormalizeCurrentRow(a, i, n);
    KEliminateColumn(a, i, n);
  }

  std::vector<float> solutions = ExtractSolutions(a, n);
  GetOutput() = solutions;

  return true;
}

bool ZyazevaSGaussJordanElSEQ::PostProcessingImpl() {
  const auto &solutions = GetOutput();
  return !solutions.empty();
}

}  // namespace zyazeva_s_gauss_jordan_elimination
