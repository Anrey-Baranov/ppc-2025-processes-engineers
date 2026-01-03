#include "tochilin_e_integration_trapezoid/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>

#include "tochilin_e_integration_trapezoid/common/include/common.hpp"

namespace tochilin_e_integration_trapezoid {

TochilinEIntegrationTrapezoidMPI::TochilinEIntegrationTrapezoidMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TochilinEIntegrationTrapezoidMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &input = GetInput();
    return input.num_intervals > 0 && input.lower_bound < input.upper_bound && input.function != nullptr;
  }
  return true;
}

bool TochilinEIntegrationTrapezoidMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &input = GetInput();
    lower_bound_ = input.lower_bound;
    upper_bound_ = input.upper_bound;
    num_intervals_ = input.num_intervals;
    function_ = input.function;
  }
  result_ = 0.0;
  return true;
}

bool TochilinEIntegrationTrapezoidMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::array<double, 3> params = {0.0, 0.0, 0.0};
  if (rank == 0) {
    params[0] = lower_bound_;
    params[1] = upper_bound_;
    params[2] = static_cast<double>(num_intervals_);
  }

  MPI_Bcast(params.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_lower = params[0];
  double local_upper = params[1];
  int total_intervals = static_cast<int>(params[2]);

  function_ = GetInput().function;

  double step = (local_upper - local_lower) / total_intervals;

  int base_intervals = total_intervals / size;
  int remainder = total_intervals % size;

  int local_start = 0;
  int local_count = 0;

  if (rank == 0) {
    local_start = 0;
    local_count = base_intervals + (rank < remainder ? 1 : 0);

    int current_pos = local_count;
    for (int proc = 1; proc < size; ++proc) {
      int count = base_intervals + (proc < remainder ? 1 : 0);
      std::array<double, 3> send_data = {static_cast<double>(current_pos), static_cast<double>(count), step};
      MPI_Send(send_data.data(), 3, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
      current_pos += count;
    }
  } else {
    std::array<double, 3> recv_data = {0.0, 0.0, 0.0};
    MPI_Recv(recv_data.data(), 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_start = static_cast<int>(recv_data[0]);
    local_count = static_cast<int>(recv_data[1]);
    step = recv_data[2];
  }

  double local_sum = 0.0;

  for (int i = local_start; i < local_start + local_count; ++i) {
    double x_left = local_lower + (i * step);
    double x_right = local_lower + ((i + 1) * step);
    local_sum += ((function_(x_left) + function_(x_right)) / 2.0) * step;
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    result_ = global_sum;
  }

  return true;
}

bool TochilinEIntegrationTrapezoidMPI::PostProcessingImpl() {
  MPI_Bcast(&result_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = result_;
  return true;
}

}  // namespace tochilin_e_integration_trapezoid
