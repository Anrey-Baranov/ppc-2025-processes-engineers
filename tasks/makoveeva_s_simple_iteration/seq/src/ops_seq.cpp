#include "makoveeva_s_simple_iteration/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "makoveeva_s_simple_iteration/common/include/common.hpp"

namespace makoveeva_s_simple_iteration {

namespace {

constexpr double kW = 0.5;
constexpr double kEps = 1e-6;
constexpr int kMaxIter = 1000;

void BuildSystem(int n, std::vector<double> *a, std::vector<double> *b) {
  const auto n_sz = static_cast<size_t>(n);
  a->assign(n_sz * n_sz, 0.0);
  b->assign(n_sz, 0.0);

  for (int i = 0; i < n; ++i) {
    const auto i_idx = static_cast<size_t>(i);

    (*a)[(i_idx * n_sz) + i_idx] = static_cast<double>(n) + 5.0;

    for (int j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      const auto j_idx = static_cast<size_t>(j);
      (*a)[(i_idx * n_sz) + j_idx] = 1.0 / (static_cast<double>(std::abs(i - j)) + 1.0);
    }

    double bi = 0.0;
    for (int j = 0; j < n; ++j) {
      const auto j_idx = static_cast<size_t>(j);
      bi += (*a)[(i_idx * n_sz) + j_idx] * static_cast<double>(j + 1);
    }
    (*b)[i_idx] = bi;
  }
}

void Iterate(int n, const std::vector<double> &a, const std::vector<double> &b, std::vector<double> *x) {
  const auto n_sz = static_cast<size_t>(n);
  std::vector<double> x_new(n_sz, 0.0);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    double err_sq = 0.0;

    for (int i = 0; i < n; ++i) {
      const auto i_idx = static_cast<size_t>(i);

      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        const auto j_idx = static_cast<size_t>(j);
        sum += a[(i_idx * n_sz) + j_idx] * (*x)[j_idx];
      }

      const double denom = a[(i_idx * n_sz) + i_idx];
      const double xi = (*x)[i_idx] + (kW * (b[i_idx] - sum) / denom);
      x_new[i_idx] = xi;

      const double diff = xi - (*x)[i_idx];
      err_sq += diff * diff;
    }

    x->swap(x_new);
    if (std::sqrt(err_sq) < kEps) {
      break;
    }
  }
}

int SumRounded(const std::vector<double> &x) {
  double sum = 0.0;
  for (double v : x) {
    sum += v;
  }
  return static_cast<int>(std::round(sum));
}

}  // namespace

MakoveevaSSimpleIterationSEQ::MakoveevaSSimpleIterationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MakoveevaSSimpleIterationSEQ::ValidationImpl() {
  return GetInput() > 0;
}

bool MakoveevaSSimpleIterationSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool MakoveevaSSimpleIterationSEQ::RunImpl() {
  const int n = GetInput();

  std::vector<double> a;
  std::vector<double> b;
  BuildSystem(n, &a, &b);

  std::vector<double> x(static_cast<size_t>(n), 0.0);
  Iterate(n, a, b, &x);

  GetOutput() = SumRounded(x);
  return true;
}

bool MakoveevaSSimpleIterationSEQ::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace makoveeva_s_simple_iteration
