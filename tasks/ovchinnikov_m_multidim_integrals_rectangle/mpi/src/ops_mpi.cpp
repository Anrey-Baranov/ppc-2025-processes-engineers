#include "ovchinnikov_m_multidim_integrals_rectangle/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "ovchinnikov_m_multidim_integrals_rectangle/common/include/common.hpp"

namespace ovchinnikov_m_multidim_integrals_rectangle {

namespace {
double DefaultFunction(const std::vector<double> &point) {
  double sum = 0.0;
  for (double coord : point) {
    sum += coord * coord;
  }
  return sum;
}
double ComputePartialIntegral(const std::function<double(const std::vector<double> &)> &func, int n, int dim,
                              const std::vector<double> &lower_bounds, const std::vector<double> &steps,
                              double cell_volume, int start_point, int end_point) {
  double partial_integral = 0.0;

  if (start_point >= end_point) {
    return partial_integral;
  }
  std::vector<int> indices(dim);
  std::vector<double> point(dim);
  for (int idx = start_point; idx < end_point; idx++) {
    int temp = idx;
    for (int i = dim - 1; i >= 0; i--) {
      indices[i] = temp % n;
      temp /= n;
    }

    for (int i = 0; i < dim; i++) {
      point[i] = lower_bounds[i] + ((indices[i] + 0.5) * steps[i]);
    }
    partial_integral += func(point) * cell_volume;
  }

  return partial_integral;
}

}  // namespace

OvchinnikovMMultiDimIntegralsRectangleMPI::OvchinnikovMMultiDimIntegralsRectangleMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool OvchinnikovMMultiDimIntegralsRectangleMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (std::get<0>(input) <= 0) {
    return false;
  }
  if (std::get<1>(input) <= 0) {
    return false;
  }

  const auto &lower_bounds = std::get<2>(input);
  const auto &upper_bounds = std::get<3>(input);
  int dim = std::get<1>(input);
  if (lower_bounds.size() != static_cast<size_t>(dim)) {
    return false;
  }
  if (upper_bounds.size() != static_cast<size_t>(dim)) {
    return false;
  }

  for (int i = 0; i < dim; i++) {
    if (lower_bounds[i] >= upper_bounds[i]) {
      return false;
    }
  }
  return true;
}

bool OvchinnikovMMultiDimIntegralsRectangleMPI::PreProcessingImpl() {
  return true;
}

bool OvchinnikovMMultiDimIntegralsRectangleMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  int n = std::get<0>(input);
  int dim = std::get<1>(input);
  const auto &lower_bounds = std::get<2>(input);
  const auto &upper_bounds = std::get<3>(input);

  std::function<double(const std::vector<double> &)> func = DefaultFunction;

  std::vector<double> steps(dim);
  double cell_volume = 1.0;

  for (int i = 0; i < dim; i++) {
    steps[i] = (upper_bounds[i] - lower_bounds[i]) / n;
    cell_volume *= steps[i];
  }

  int total_points = static_cast<int>(std::pow(n, dim));

  double local_integral = 0.0;
  double global_integral = 0.0;

  if (rank == 0) {
    std::array<int, 2> params = {n, dim};
    MPI_Bcast(params.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> all_bounds;
    all_bounds.insert(all_bounds.end(), lower_bounds.begin(), lower_bounds.end());
    all_bounds.insert(all_bounds.end(), upper_bounds.begin(), upper_bounds.end());

    MPI_Bcast(all_bounds.data(), 2 * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cell_volume, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int points_per_process = total_points / size;
    int remainder = total_points % size;

    int start_point = 0;
    int extra_points = static_cast<int>(remainder > 0);
    int end_point = points_per_process + extra_points;
    local_integral = ComputePartialIntegral(func, n, dim, lower_bounds, steps, cell_volume, start_point, end_point);

    for (int i = 1; i < size; i++) {
      start_point = 0;
      for (int j = 0; j < i; j++) {
        int extra_points_for_j = static_cast<int>(j < remainder);
        int points_for_j = points_per_process + extra_points_for_j;
        start_point += points_for_j;
      }
      extra_points = static_cast<int>(i < remainder);
      end_point = start_point + points_per_process + extra_points;

      std::array<int, 2> range = {start_point, end_point};
      MPI_Send(range.data(), 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    global_integral = local_integral;

    for (int i = 1; i < size; i++) {
      double partial_result = 0.0;
      MPI_Recv(&partial_result, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_integral += partial_result;
    }

    GetOutput() = global_integral;

    MPI_Bcast(&global_integral, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  if (rank != 0) {
    std::array<int, 2> params{{0, 0}};
    MPI_Bcast(params.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
    n = params[0];
    dim = params[1];

    std::vector<double> all_bounds(static_cast<size_t>(2) * dim);
    MPI_Bcast(all_bounds.data(), 2 * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> local_lower_bounds(all_bounds.begin(), all_bounds.begin() + dim);
    std::vector<double> local_upper_bounds(all_bounds.begin() + dim, all_bounds.end());

    MPI_Bcast(&cell_volume, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> local_steps(dim);
    for (int i = 0; i < dim; i++) {
      local_steps[i] = (local_upper_bounds[i] - local_lower_bounds[i]) / n;
    }

    std::array<int, 2> range{{0, 0}};
    MPI_Recv(range.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int start_point = range[0];
    int end_point = range[1];

    local_integral =
        ComputePartialIntegral(func, n, dim, local_lower_bounds, local_steps, cell_volume, start_point, end_point);

    MPI_Send(&local_integral, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

    MPI_Bcast(&global_integral, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    GetOutput() = global_integral;
  }

  return true;
}

bool OvchinnikovMMultiDimIntegralsRectangleMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ovchinnikov_m_multidim_integrals_rectangle
