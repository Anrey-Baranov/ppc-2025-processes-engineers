#include "samoylenko_i_simple_iter_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "samoylenko_i_simple_iter_method/common/include/common.hpp"

namespace samoylenko_i_simple_iter_method {

SamoylenkoISimpleIterMethodSEQ::SamoylenkoISimpleIterMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SamoylenkoISimpleIterMethodSEQ::ValidationImpl() {
  return (GetInput() > 0) && GetOutput().empty();
}

bool SamoylenkoISimpleIterMethodSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

std::vector<double> BuildMatrix(size_t size) {
  std::vector<double> matrix(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    matrix[(i * size) + i] = 4.0;
    if (i > 0) {
      matrix[(i * size) + (i - 1)] = 1.0;
    }
    if (i + 1 < size) {
      matrix[(i * size) + (i + 1)] = 1.0;
    }
  }
  return matrix;
}

void PerformIterations(size_t size, const std::vector<double> &matrix, const std::vector<double> &vector,
                       std::vector<double> &x_new, std::vector<double> &x_old) {
  const double tau = 0.2;
  const double eps = 1e-7;
  const int iters = 2000;

  for (int it = 0; it < iters; ++it) {
    for (size_t i = 0; i < size; ++i) {
      double ax_i = 0.0;
      for (size_t j = 0; j < size; ++j) {
        ax_i += matrix[(i * size) + j] * x_old[j];
      }
      x_new[i] = x_old[i] - (tau * (ax_i - vector[i]));
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < size; ++i) {
      max_diff = std::max(max_diff, std::fabs(x_new[i] - x_old[i]));
    }

    x_old.swap(x_new);

    if (max_diff < eps) {
      break;
    }
  }
}

}  // namespace

bool SamoylenkoISimpleIterMethodSEQ::RunImpl() {
  const int n = GetInput();
  if (n <= 0) {
    return false;
  }

  auto size = static_cast<size_t>(n);

  std::vector<double> matrix = BuildMatrix(size);
  std::vector<double> vector(size, 1.0);

  std::vector<double> x_old(size, 0.0);
  std::vector<double> x_new(size);

  PerformIterations(size, matrix, vector, x_new, x_old);

  GetOutput() = x_old;
  return true;
}

bool SamoylenkoISimpleIterMethodSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace samoylenko_i_simple_iter_method
