#include "shkryleva_s_seidel_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "shkryleva_s_seidel_method/common/include/common.hpp"

namespace shkryleva_s_seidel_method {

ShkrylevaSSeidelMethodSEQ::ShkrylevaSSeidelMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkrylevaSSeidelMethodSEQ::ValidationImpl() {
  n_ = GetInput();
  return (n_ > 0);
}

bool ShkrylevaSSeidelMethodSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ShkrylevaSSeidelMethodSEQ::RunImpl() {
  GenerateRandomMatrix(n_, A_, b_);

  ComputeRightHandSide(n_, A_, b_);

  epsilon_ = 1e-6;
  max_iterations_ = 10000;
  x_.assign(n_, 0.0);

  int iteration = 0;
  bool converged = false;

  while (iteration < max_iterations_ && !converged) {
    double max_diff = PerformSeidelIteration(n_, A_, b_, x_);
    converged = (max_diff < epsilon_);
    ++iteration;
  }

  double sum = std::accumulate(x_.begin(), x_.end(), 0.0);
  GetOutput() = static_cast<int>(std::round(sum));

  return true;
}

bool ShkrylevaSSeidelMethodSEQ::PostProcessingImpl() {
  return true;
}

void ShkrylevaSSeidelMethodSEQ::GenerateRandomMatrix(int size, std::vector<std::vector<double>> &matrix,
                                                     std::vector<double> &vector) {
  matrix.resize(size);
  for (int i = 0; i < size; ++i) {
    matrix[i].assign(size, 0.0);
  }
  vector.resize(size, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 10);
  std::uniform_int_distribution<> dist_diag(1, 5);

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < size; ++j) {
      if (i != j) {
        matrix[i][j] = static_cast<double>(dist(gen));
        row_sum += std::abs(matrix[i][j]);
      }
    }

    matrix[i][i] = row_sum + static_cast<double>(dist_diag(gen));
  }
}

void ShkrylevaSSeidelMethodSEQ::ComputeRightHandSide(int n, const std::vector<std::vector<double>> &a,
                                                     std::vector<double> &b) {
  std::vector<double> x_exact(n, 1.0);
  b.assign(n, 0.0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      b[i] += a[i][j] * x_exact[j];
    }
  }
}

double ShkrylevaSSeidelMethodSEQ::PerformSeidelIteration(int n, const std::vector<std::vector<double>> &a,
                                                         const std::vector<double> &b, std::vector<double> &x) {
  double max_diff = 0.0;

  for (int i = 0; i < n; ++i) {
    const double old = x[i];
    double sum_off_diag = 0.0;

    for (int j = 0; j < n; ++j) {
      if (i != j) {
        sum_off_diag += a[i][j] * x[j];
      }
    }

    x[i] = (b[i] - sum_off_diag) / a[i][i];
    const double diff = std::abs(x[i] - old);
    max_diff = std::max(diff, max_diff);
  }

  return max_diff;
}

}  // namespace shkryleva_s_seidel_method
