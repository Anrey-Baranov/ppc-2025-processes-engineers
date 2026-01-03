#include "samoylenko_i_conj_grad_method/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "samoylenko_i_conj_grad_method/common/include/common.hpp"

namespace samoylenko_i_conj_grad_method {

SamoylenkoIConjGradMethodSEQ::SamoylenkoIConjGradMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SamoylenkoIConjGradMethodSEQ::ValidationImpl() {
  return (GetInput().first > 0) && (GetInput().second >= 0 && GetInput().second <= 2) && GetOutput().empty();
}

bool SamoylenkoIConjGradMethodSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

std::vector<double> BuildMatrix(size_t size, int variant) {
  std::vector<double> matrix(size * size, 0.0);

  for (size_t i = 0; i < size; ++i) {
    switch (variant) {
      case 0: {
        matrix[(i * size) + i] = 4.0;
        if (i > 0) {
          matrix[(i * size) + (i - 1)] = 1.0;
        }
        if (i + 1 < size) {
          matrix[(i * size) + (i + 1)] = 1.0;
        }
        break;
      }

      case 1: {
        matrix[(i * size) + i] = 5.0;
        break;
      }

      case 2: {
        matrix[(i * size) + i] = 3.0;
        size_t j = size - 1 - i;
        if (i != j) {
          matrix[(i * size) + j] = -1.0;
          matrix[(j * size) + i] = -1.0;
        }
        break;
      }

      default:
        break;
    }
  }
  return matrix;
}

void MatrixVectorMult(size_t size, const std::vector<double> &matrix, const std::vector<double> &vec,
                      std::vector<double> &result) {
  for (size_t i = 0; i < size; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < size; ++j) {
      sum += matrix[(i * size) + j] * vec[j];
    }

    result[i] = sum;
  }
}

double DotProduct(const std::vector<double> &first, const std::vector<double> &second) {
  double sum = 0.0;
  for (size_t i = 0; i < first.size(); ++i) {
    sum += first[i] * second[i];
  }

  return sum;
}

void ConjugateGradient(size_t size, const std::vector<double> &matrix, const std::vector<double> &vector,
                       std::vector<double> &x) {
  const double eps = 1e-7;
  const int iters = 2000;

  std::vector<double> res(size);
  std::vector<double> dir(size);
  std::vector<double> matdir(size);

  MatrixVectorMult(size, matrix, x, res);
  for (size_t i = 0; i < size; ++i) {
    res[i] = vector[i] - res[i];
    dir[i] = res[i];
  }

  double res_dot = DotProduct(res, res);

  for (int it = 0; it < iters; ++it) {
    if (std::sqrt(res_dot) < eps) {
      break;
    }

    MatrixVectorMult(size, matrix, dir, matdir);

    double matdir_dot = DotProduct(dir, matdir);
    if (std::fabs(matdir_dot) < 1e-15) {
      break;  // so we dont divide by 0
    }

    double step = res_dot / matdir_dot;

    for (size_t i = 0; i < size; ++i) {
      x[i] += step * dir[i];
      res[i] -= step * matdir[i];
    }

    double res_dot_new = DotProduct(res, res);
    double conj_coef = res_dot_new / res_dot;

    for (size_t i = 0; i < size; ++i) {
      dir[i] = res[i] + (conj_coef * dir[i]);
    }

    res_dot = res_dot_new;
  }
}

}  // namespace

bool SamoylenkoIConjGradMethodSEQ::RunImpl() {
  const int n = GetInput().first;
  const int variant = GetInput().second;
  if (n <= 0) {
    return false;
  }

  auto size = static_cast<size_t>(n);

  std::vector<double> matrix = BuildMatrix(size, variant);
  std::vector<double> vector(size, 1.0);

  std::vector<double> x(size, 0.0);

  ConjugateGradient(size, matrix, vector, x);

  GetOutput() = x;
  return true;
}

bool SamoylenkoIConjGradMethodSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace samoylenko_i_conj_grad_method
