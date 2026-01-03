#include "Rastvorov_K_Simple_iteration_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "Rastvorov_K_Simple_iteration_method/common/include/common.hpp"

namespace rastvorov_k_simple_iteration_method {

namespace {

inline void ComputeRange(std::size_t rank, std::size_t size, std::size_t total, std::size_t &begin, std::size_t &end) {
  const std::size_t base = total / size;
  const std::size_t rem = total % size;

  if (rank < rem) {
    begin = rank * (base + 1);
    end = begin + base + 1;
  } else {
    begin = (rem * (base + 1)) + ((rank - rem) * base);
    end = begin + base;
  }
}

inline std::vector<double> RunIterations(int n, std::size_t local_n, std::vector<double> x_local) {
  constexpr double kEps = 1e-9;
  constexpr int kMaxIter = 2000;

  std::vector<double> x_new_local(local_n, 0.0);
  const auto denom = static_cast<double>(2 * n);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    double local_sum = 0.0;
    for (double v : x_local) {
      local_sum += v;
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double local_max_diff = 0.0;
    for (std::size_t idx = 0; idx < local_n; ++idx) {
      const double old = x_local[idx];
      const double xnew = (1.0 - global_sum + old) / denom;
      x_new_local[idx] = xnew;
      local_max_diff = std::max(local_max_diff, std::abs(xnew - old));
    }

    x_local.swap(x_new_local);

    double global_max_diff = 0.0;
    MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (global_max_diff < kEps) {
      break;
    }
  }

  return x_local;
}

}  // namespace

RastvorovKSimpleIterationMethodMPI::RastvorovKSimpleIterationMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RastvorovKSimpleIterationMethodMPI::ValidationImpl() {
  return GetInput() > 0;
}

bool RastvorovKSimpleIterationMethodMPI::PreProcessingImpl() {
  GetOutput().assign(static_cast<std::size_t>(std::max(GetInput(), 0)), 0.0);
  return true;
}

bool RastvorovKSimpleIterationMethodMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = GetInput();
  if (n <= 0) {
    if (rank == 0) {
      GetOutput().clear();
    }
    return false;
  }

  const auto total = static_cast<std::size_t>(n);

  std::size_t begin = 0;
  std::size_t end = 0;
  ComputeRange(static_cast<std::size_t>(rank), static_cast<std::size_t>(size), total, begin, end);
  const std::size_t local_n = (end > begin) ? (end - begin) : 0;

  std::vector<double> x_local(local_n, 0.0);
  x_local = RunIterations(n, local_n, std::move(x_local));

  std::vector<int> recvcounts;
  std::vector<int> displs;
  std::vector<double> x_full;

  if (rank == 0) {
    recvcounts.resize(static_cast<std::size_t>(size));
    displs.resize(static_cast<std::size_t>(size));

    for (int proc = 0; proc < size; ++proc) {
      std::size_t b = 0;
      std::size_t e = 0;
      ComputeRange(static_cast<std::size_t>(proc), static_cast<std::size_t>(size), total, b, e);
      recvcounts[static_cast<std::size_t>(proc)] = static_cast<int>(e - b);
      displs[static_cast<std::size_t>(proc)] = static_cast<int>(b);
    }

    x_full.assign(total, 0.0);
  }

  MPI_Gatherv(x_local.data(), static_cast<int>(local_n), MPI_DOUBLE, rank == 0 ? x_full.data() : nullptr,
              rank == 0 ? recvcounts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = x_full;
  }

  return true;
}

bool RastvorovKSimpleIterationMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rastvorov_k_simple_iteration_method
