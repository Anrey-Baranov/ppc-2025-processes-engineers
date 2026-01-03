#include "Rastvorov_K_Simple_iteration_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "Rastvorov_K_Simple_iteration_method/common/include/common.hpp"

namespace rastvorov_k_simple_iteration_method {

namespace {}  // namespace

RastvorovKSimpleIterationMethodSEQ::RastvorovKSimpleIterationMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RastvorovKSimpleIterationMethodSEQ::ValidationImpl() {
  return GetInput() > 0;
}

bool RastvorovKSimpleIterationMethodSEQ::PreProcessingImpl() {
  GetOutput().assign(static_cast<std::size_t>(GetInput()), 0.0);
  return true;
}

bool RastvorovKSimpleIterationMethodSEQ::RunImpl() {
  const int n = GetInput();
  if (n <= 0) {
    GetOutput().clear();
    return false;
  }

  constexpr double kEps = 1e-9;
  constexpr int kMaxIter = 2000;

  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  std::vector<double> x_new(static_cast<std::size_t>(n), 0.0);

  const auto denom = static_cast<double>(2 * n);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    double sum_all = 0.0;
    for (double v : x) {
      sum_all += v;
    }

    double max_diff = 0.0;

    for (int i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      const double old = x[idx];
      const double xnew = (1.0 - sum_all + old) / denom;
      x_new[idx] = xnew;
      max_diff = std::max(max_diff, std::abs(xnew - old));
    }

    x.swap(x_new);

    if (max_diff < kEps) {
      break;
    }
  }

  GetOutput() = x;
  return true;
}

bool RastvorovKSimpleIterationMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rastvorov_k_simple_iteration_method
