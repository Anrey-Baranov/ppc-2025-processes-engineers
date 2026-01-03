#include "afanasyev_a_it_seidel_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "afanasyev_a_it_seidel_method/common/include/common.hpp"

namespace afanasyev_a_it_seidel_method {

namespace {

double CalculateMaxDiff(const std::vector<double> &a, const std::vector<double> &b) {
  double max_diff = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  }
  return max_diff;
}

void SafeVectorCopy(std::vector<double> &dest, const std::vector<double> &src) {
  if (dest.size() == src.size()) {
    for (std::size_t i = 0; i < src.size(); ++i) {
      dest[i] = src[i];
    }
  }
}

bool PerformIteration(int system_size, int start_row, int end_row, const std::vector<std::vector<double>> &a,
                      const std::vector<double> &b, std::vector<double> &local_x, std::vector<double> &global_x) {
  for (int i = start_row; i < end_row; ++i) {
    if (i >= system_size) {
      break;
    }

    double sum = b[i];

    for (int j = 0; j < i; ++j) {
      sum -= a[i][j] * global_x[j];
    }

    for (int j = i + 1; j < system_size; ++j) {
      sum -= a[i][j] * global_x[j];
    }

    local_x[i] = sum / a[i][i];
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_process = system_size / size;
  int remainder = system_size % size;

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  for (int proc = 0; proc < size; ++proc) {
    int cnt = rows_per_process + (proc < remainder ? 1 : 0);
    sendcounts[proc] = cnt;
    displs[proc] = (proc * rows_per_process) + std::min(proc, remainder);
  }

  int sendcount = end_row - start_row;
  MPI_Allgatherv(local_x.data() + start_row, sendcount, MPI_DOUBLE, global_x.data(), sendcounts.data(), displs.data(),
                 MPI_DOUBLE, MPI_COMM_WORLD);

  return true;
}

bool CheckConvergence(int rank, double max_diff, double epsilon, const std::vector<double> &global_x,
                      std::vector<double> &x) {
  if (rank == 0) {
    if (max_diff < epsilon) {
      int converged = 1;
      MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
      SafeVectorCopy(x, global_x);
      return true;
    }

    int converged = 0;
    MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    int converged = 0;
    MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (converged != 0) {
      return true;
    }
  }

  return false;
}
}  // namespace

AfanasyevAItSeidelMethodMPI::AfanasyevAItSeidelMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool AfanasyevAItSeidelMethodMPI::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool AfanasyevAItSeidelMethodMPI::PreProcessingImpl() {
  try {
    int system_size = static_cast<int>(GetInput()[0]);
    epsilon_ = GetInput()[1];
    max_iterations_ = static_cast<int>(GetInput()[2]);

    A_.clear();
    A_.resize(system_size);
    for (int i = 0; i < system_size; ++i) {
      A_[i].resize(system_size);
      for (int j = 0; j < system_size; ++j) {
        if (i == j) {
          A_[i][j] = system_size + 1.0;
        } else {
          A_[i][j] = 1.0 / (std::abs(i - j) + 1.0);
        }
      }
    }

    b_.clear();
    b_.resize(system_size);
    for (int i = 0; i < system_size; ++i) {
      b_[i] = i + 1.0;
    }

    x_.clear();
    x_.resize(system_size, 0.0);

    return true;
  } catch (...) {
    return false;
  }
}

bool AfanasyevAItSeidelMethodMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int system_size = static_cast<int>(A_.size());
  if (system_size == 0) {
    return false;
  }

  int rows_per_process = system_size / size;
  int remainder = system_size % size;

  int start_row = (rank * rows_per_process) + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

  std::vector<double> local_x(system_size, 0.0);
  std::vector<double> global_x(system_size, 0.0);

  for (int iter = 0; iter < max_iterations_; ++iter) {
    std::vector<double> prev_x = global_x;

    if (!PerformIteration(system_size, start_row, end_row, A_, b_, local_x, global_x)) {
      return false;
    }

    if (CheckConvergence(rank, CalculateMaxDiff(global_x, prev_x), epsilon_, global_x, x_)) {
      break;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    OutType output;
    output.reserve(x_.size());
    for (const double val : x_) {
      output.push_back(val);
    }
    GetOutput() = output;
  } else {
    GetOutput() = std::vector<double>();
  }
  // Synchronize output on all ranks so tests running per-process can validate output
  {
    int out_size = static_cast<int>(GetOutput().size());
    MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput().resize(out_size);
    if (out_size > 0) {
      MPI_Bcast(GetOutput().data(), out_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  }

  return true;
}

bool AfanasyevAItSeidelMethodMPI::PostProcessingImpl() {
  try {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    int system_size = static_cast<int>(A_.size());
    if (x_.size() != static_cast<std::size_t>(system_size)) {
      return false;
    }

    double residual_norm = 0.0;
    for (int i = 0; i < system_size; ++i) {
      double sum = 0.0;
      for (int j = 0; j < system_size; ++j) {
        sum += A_[i][j] * x_[j];
      }
      residual_norm += std::abs(sum - b_[i]);
    }

    residual_norm /= system_size;
    return residual_norm < epsilon_ * 1000;
  } catch (...) {
    return false;
  }
}

}  // namespace afanasyev_a_it_seidel_method
