#include "timofeev_n_ribbon_scheme_only_a/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "timofeev_n_ribbon_scheme_only_a/common/include/common.hpp"

namespace timofeev_n_ribbon_scheme_only_a {

TimofeevNRibbonSchemeOnlyASEQ::TimofeevNRibbonSchemeOnlyASEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<std::vector<int>>(0, std::vector<int>(0));
}

bool TimofeevNRibbonSchemeOnlyASEQ::ValidationImpl() {
  std::vector<std::vector<int>> &a = GetInput().first;
  std::vector<std::vector<int>> &b = GetInput().second;
  size_t n = a.size();
  size_t m2 = a[0].size();
  size_t m1 = b.size();
  size_t k = b[0].size();
  return (m2 == m1 && n != 0 && m1 != 0 && m2 != 0 && k != 0);
}

bool TimofeevNRibbonSchemeOnlyASEQ::PreProcessingImpl() {
  return true;
}

int TimofeevNRibbonSchemeOnlyASEQ::CalculatingCElement(MatrixType &a, MatrixType &b, size_t &i, size_t &j) {
  int summ = 0;
  for (size_t kk = 0; kk < a[0].size(); kk++) {
    summ += a[i][kk] * b[kk][j];
  }
  return summ;
}

bool TimofeevNRibbonSchemeOnlyASEQ::RunImpl() {
  MatrixType a = GetInput().first;
  MatrixType b = GetInput().second;
  std::vector<std::vector<int>> c(a.size(), std::vector<int>(b[0].size()));
  for (size_t i = 0; i < a.size(); i++) {
    for (size_t j = 0; j < b[0].size(); j++) {
      c[i][j] = CalculatingCElement(a, b, i, j);
    }
  }
  GetOutput() = c;
  return true;
}

bool TimofeevNRibbonSchemeOnlyASEQ::PostProcessingImpl() {
  return true;
}

}  // namespace timofeev_n_ribbon_scheme_only_a
