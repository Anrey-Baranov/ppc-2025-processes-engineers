#include "zaharov_g_seidel_int_met/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "util/include/util.hpp"
#include "zaharov_g_seidel_int_met/common/include/common.hpp"

namespace zaharov_g_seidel_int_met {

ZaharovGSeidelIntMetSEQ::ZaharovGSeidelIntMetSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool ZaharovGSeidelIntMetSEQ::ValidationImpl() {
  const InType &input = GetInput();
  if (input.size() < 3) {
    return false;
  }

  if (input[0] <= 0 || input[1] <= 0.0 || input[2] <= 0) {
    return false;
  }

  int system_size = static_cast<int>(input[0]);
  int max_iterations = static_cast<int>(input[2]);

  if (static_cast<double>(system_size) != input[0] || static_cast<double>(max_iterations) != input[2]) {
    return false;
  }

  return true;
}

bool ZaharovGSeidelIntMetSEQ::PreProcessingImpl() {
  try {
    const int system_size = static_cast<int>(GetInput()[0]);
    epsilon_ = GetInput()[1];
    max_iterations_ = static_cast<int>(GetInput()[2]);

    A_.resize(system_size);
    b_.resize(system_size);
    x_.assign(system_size, 0.0);

    for (int i = 0; i < system_size; ++i) {
      A_[i].resize(system_size);
      double *row = A_[i].data();

      row[i] = system_size + 1.0;

      for (int j = 0; j < i; ++j) {
        row[j] = 1.0 / (i - j + 1.0);
      }

      for (int j = i + 1; j < system_size; ++j) {
        row[j] = 1.0 / (j - i + 1.0);
      }

      b_[i] = static_cast<double>(i + 1);
    }

    return true;
  } catch (...) {
    return false;
  }
}

bool ZaharovGSeidelIntMetSEQ::RunImpl() {
  try {
    const int system_size = static_cast<int>(A_.size());
    if (system_size == 0) {
      return false;
    }

    std::vector<double> old_x(system_size);

    for (int iter = 0; iter < max_iterations_; ++iter) {
      std::copy(x_.begin(), x_.end(), old_x.begin());
      double max_diff = 0.0;

      for (int i = 0; i < system_size; ++i) {
        double sum = b_[i];

        for (int j = 0; j < i; ++j) {
          sum -= A_[i][j] * x_[j];
        }

        for (int j = i + 1; j < system_size; ++j) {
          sum -= A_[i][j] * old_x[j];
        }

        x_[i] = sum / A_[i][i];
        const double diff = std::abs(x_[i] - old_x[i]);
        if (diff > max_diff) {
          max_diff = diff;
        }
      }

      if (max_diff < epsilon_) {
        break;
      }
    }

    GetOutput() = OutType(x_.begin(), x_.end());
    return true;
  } catch (...) {
    return false;
  }
}

bool ZaharovGSeidelIntMetSEQ::PostProcessingImpl() {
  try {
    const int system_size = static_cast<int>(A_.size());
    if (x_.size() != static_cast<std::size_t>(system_size)) {
      return false;
    }

    double residual_norm = 0.0;
    for (int i = 0; i < system_size; ++i) {
      double sum = 0.0;

      const auto &row = A_[i];
      for (int j = 0; j < system_size; ++j) {
        sum += row[j] * x_[j];
      }

      residual_norm += std::abs(sum - b_[i]);
    }

    double b_norm = 0.0;
    for (const auto &bi : b_) {
      b_norm += std::abs(bi);
    }

    return (residual_norm / (b_norm + std::numeric_limits<double>::epsilon())) < epsilon_;
  } catch (...) {
    return false;
  }
}

}  // namespace zaharov_g_seidel_int_met
