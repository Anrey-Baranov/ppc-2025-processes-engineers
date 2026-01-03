#include "zaharov_g_seidel_int_met/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "zaharov_g_seidel_int_met/common/include/common.hpp"

namespace zaharov_g_seidel_int_met {

ZaharovGSeidelIntMetMPI::ZaharovGSeidelIntMetMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool ZaharovGSeidelIntMetMPI::ValidationImpl() {
  const InType &input = GetInput();
  if (input.size() < 3) {
    return false;
  }

  if (input[0] <= 0 || input[1] <= 0.0 || input[2] <= 0) {
    return false;
  }

  const int system_size = static_cast<int>(input[0]);
  const int max_iterations = static_cast<int>(input[2]);

  return (static_cast<double>(system_size) == input[0]) && (static_cast<double>(max_iterations) == input[2]);
}

bool ZaharovGSeidelIntMetMPI::PreProcessingImpl() {
  try {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    system_size_ = static_cast<int>(GetInput()[0]);
    epsilon_ = GetInput()[1];
    max_iterations_ = static_cast<int>(GetInput()[2]);

    rows_per_process_ = system_size_ / size_;
    remainder_ = system_size_ % size_;

    start_row_ = (rank_ * rows_per_process_) + std::min(rank_, remainder_);
    end_row_ = start_row_ + rows_per_process_ + (rank_ < remainder_ ? 1 : 0);
    local_rows_ = end_row_ - start_row_;

    local_A_.resize(local_rows_);
    local_b_.resize(local_rows_);

    for (int local_i = 0; local_i < local_rows_; ++local_i) {
      const int global_i = start_row_ + local_i;
      local_A_[local_i].resize(system_size_);

      for (int j = 0; j < system_size_; ++j) {
        if (global_i == j) {
          local_A_[local_i][j] = system_size_ + 1.0;
        } else {
          local_A_[local_i][j] = 1.0 / (static_cast<double>(std::abs(global_i - j)) + 1.0);
        }
      }

      local_b_[local_i] = static_cast<double>(global_i + 1);
    }

    x_.resize(system_size_, 0.0);

    return true;
  } catch (...) {
    return false;
  }
}

bool ZaharovGSeidelIntMetMPI::RunImpl() {
  if (system_size_ == 0) {
    return false;
  }

  std::vector<double> prev_x(system_size_, 0.0);
  std::vector<double> local_x_new(local_rows_, 0.0);

  std::vector<int> recvcounts(size_);
  std::vector<int> displs(size_);

  for (int i = 0; i < size_; ++i) {
    const int i_start = (i * rows_per_process_) + std::min(i, remainder_);
    const int i_end = i_start + rows_per_process_ + (i < remainder_ ? 1 : 0);
    recvcounts[i] = i_end - i_start;
    displs[i] = i_start;
  }

  for (int iter = 0; iter < max_iterations_; ++iter) {
    std::ranges::copy(x_, prev_x.begin());

    for (int local_i = 0; local_i < local_rows_; ++local_i) {
      const int global_i = start_row_ + local_i;
      double sum = local_b_[local_i];
      const auto &row = local_A_[local_i];

      for (int j = 0; j < global_i; ++j) {
        sum -= row[j] * x_[j];
      }

      for (int j = global_i + 1; j < system_size_; ++j) {
        sum -= row[j] * prev_x[j];
      }

      local_x_new[local_i] = sum / row[global_i];
    }

    MPI_Allgatherv(local_x_new.data(), local_rows_, MPI_DOUBLE, x_.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    double local_max_diff = 0.0;
    for (int local_i = 0; local_i < local_rows_; ++local_i) {
      const int global_i = start_row_ + local_i;
      const double diff = std::abs(x_[global_i] - prev_x[global_i]);
      local_max_diff = std::max(diff, local_max_diff);
    }

    double global_max_diff = 0.0;
    MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (global_max_diff < epsilon_) {
      break;
    }
  }

  GetOutput() = std::vector<double>(x_.begin(), x_.end());

  return true;
}

bool ZaharovGSeidelIntMetMPI::PostProcessingImpl() {
  try {
    if (rank_ != 0) {
      return true;
    }

    if (x_.size() != static_cast<std::size_t>(system_size_)) {
      return false;
    }

    std::vector<std::vector<double>> matrix_a(system_size_, std::vector<double>(system_size_));
    std::vector<double> vector_b(system_size_);

    for (int i = 0; i < system_size_; ++i) {
      vector_b[i] = static_cast<double>(i + 1);
      for (int j = 0; j < system_size_; ++j) {
        if (i == j) {
          matrix_a[i][j] = system_size_ + 1.0;
        } else {
          matrix_a[i][j] = 1.0 / (static_cast<double>(std::abs(i - j)) + 1.0);
        }
      }
    }

    double residual_norm = 0.0;
    for (int i = 0; i < system_size_; ++i) {
      double sum = 0.0;
      for (int j = 0; j < system_size_; ++j) {
        sum += matrix_a[i][j] * x_[j];
      }
      residual_norm += std::abs(sum - vector_b[i]);
    }

    residual_norm /= static_cast<double>(system_size_);

    return residual_norm < (epsilon_ * 1000.0);
  } catch (...) {
    return false;
  }
}

}  // namespace zaharov_g_seidel_int_met
