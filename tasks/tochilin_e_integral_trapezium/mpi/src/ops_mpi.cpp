#include "tochilin_e_integral_trapezium/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "tochilin_e_integral_trapezium/common/include/common.hpp"

namespace tochilin_e_integral_trapezium {

TochilinEIntegralTrapeziumMPI::TochilinEIntegralTrapeziumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TochilinEIntegralTrapeziumMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();

    if (input.lower_bounds.empty() || input.upper_bounds.empty()) {
      return false;
    }

    if (input.lower_bounds.size() != input.upper_bounds.size()) {
      return false;
    }

    if (input.num_steps <= 0) {
      return false;
    }

    if (!input.func) {
      return false;
    }

    for (std::size_t idx = 0; idx < input.lower_bounds.size(); ++idx) {
      if (input.lower_bounds[idx] > input.upper_bounds[idx]) {
        return false;
      }
    }
  }

  return true;
}

bool TochilinEIntegralTrapeziumMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dimensions = 0;
  if (rank == 0) {
    const auto &input = GetInput();
    lower_bounds_ = input.lower_bounds;
    upper_bounds_ = input.upper_bounds;
    num_steps_ = input.num_steps;
    func_ = input.func;
    dimensions = static_cast<int>(lower_bounds_.size());
  }

  MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_steps_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    lower_bounds_.resize(dimensions);
    upper_bounds_.resize(dimensions);
    func_ = GetInput().func;
  }

  MPI_Bcast(lower_bounds_.data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(upper_bounds_.data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int total_points = 1;
  for (int idx = 0; idx < dimensions; ++idx) {
    total_points *= (num_steps_ + 1);
  }

  int base_points_per_proc = total_points / size;
  int remainder = total_points % size;

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  int offset = 0;
  for (int idx = 0; idx < size; ++idx) {
    counts[idx] = base_points_per_proc + (idx < remainder ? 1 : 0);
    displs[idx] = offset;
    offset += counts[idx];
  }

  if (rank == 0) {
    for (int dest = 1; dest < size; ++dest) {
      MPI_Send(&counts[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&displs[dest], 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
    }
    start_idx_ = displs[0];
    end_idx_ = displs[0] + counts[0];
  } else {
    int local_count = 0;
    int local_displ = 0;
    MPI_Recv(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&local_displ, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    start_idx_ = local_displ;
    end_idx_ = local_displ + local_count;
  }

  result_ = 0.0;
  return true;
}

double TochilinEIntegralTrapeziumMPI::ComputePartialIntegral(int start_idx, int end_idx) {
  std::size_t dimensions = lower_bounds_.size();
  std::vector<double> step_sizes(dimensions);
  for (std::size_t idx = 0; idx < dimensions; ++idx) {
    step_sizes[idx] = (upper_bounds_[idx] - lower_bounds_[idx]) / num_steps_;
  }

  double sum = 0.0;
  std::vector<double> point(dimensions);

  for (int idx = start_idx; idx < end_idx; ++idx) {
    int temp = idx;
    double weight = 1.0;

    for (std::size_t dim = 0; dim < dimensions; ++dim) {
      int grid_idx = temp % (num_steps_ + 1);
      temp /= (num_steps_ + 1);
      point[dim] = lower_bounds_[dim] + (grid_idx * step_sizes[dim]);

      if (grid_idx == 0 || grid_idx == num_steps_) {
        weight *= 0.5;
      }
    }

    sum += weight * func_(point);
  }

  double volume = 1.0;
  for (std::size_t idx = 0; idx < dimensions; ++idx) {
    volume *= step_sizes[idx];
  }

  return sum * volume;
}

bool TochilinEIntegralTrapeziumMPI::RunImpl() {
  double local_result = ComputePartialIntegral(start_idx_, end_idx_);

  double global_result = 0.0;
  MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    result_ = global_result;
  }

  return true;
}

bool TochilinEIntegralTrapeziumMPI::PostProcessingImpl() {
  MPI_Bcast(&result_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = result_;
  return true;
}

}  // namespace tochilin_e_integral_trapezium
