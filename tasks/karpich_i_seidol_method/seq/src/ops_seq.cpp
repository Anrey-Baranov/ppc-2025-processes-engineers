#include "karpich_i_seidol_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "karpich_i_seidol_method/common/include/common.hpp"

namespace karpich_i_seidol_method {

KarpichISeidolMethodSEQ::KarpichISeidolMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KarpichISeidolMethodSEQ::ValidationImpl() {
  return true;
}

bool KarpichISeidolMethodSEQ::PreProcessingImpl() {
  return true;
}

bool KarpichISeidolMethodSEQ::RunImpl() {
  std::size_t n = std::get<0>(GetInput());
  std::vector<double> &a = std::get<1>(GetInput());
  std::vector<double> &b = std::get<2>(GetInput());
  double eps = std::get<3>(GetInput());

  std::vector<double> x(n, 0);
  std::vector<double> epsilons(n, -1);
  bool iter_continue = true;
  while (iter_continue) {
    for (std::size_t i = 0; i < n; i++) {
      double ix = b[i];
      for (std::size_t j = 0; j < i; j++) {
        ix = ix - (a[(i * n) + j] * x[j]);
      }
      for (std::size_t j = i + 1; j < n; j++) {
        ix = ix - (a[(i * n) + j] * x[j]);
      }
      ix = ix / a[(i * n) + i];
      epsilons[i] = std::fabs(ix - x[i]);
      x[i] = ix;
    }
    iter_continue = IterContinue(epsilons, eps);
  }
  GetOutput() = x;
  return true;
}

bool KarpichISeidolMethodSEQ::PostProcessingImpl() {
  return true;
}

bool KarpichISeidolMethodSEQ::IterContinue(std::vector<double> &iter_eps, double correct_eps) {
  double max_in_iter = *std::ranges::max_element(iter_eps);
  return max_in_iter > correct_eps;
}

}  // namespace karpich_i_seidol_method
