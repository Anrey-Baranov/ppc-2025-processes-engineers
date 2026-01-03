#include "ovchinnikov_m_multidim_integrals_rectangle/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "ovchinnikov_m_multidim_integrals_rectangle/common/include/common.hpp"

namespace ovchinnikov_m_multidim_integrals_rectangle {

namespace {
double MultivariableFunction(const std::vector<double> &point) {
  double sum = 0.0;
  for (double coord : point) {
    sum += coord * coord;
  }
  return sum;
}
}  // namespace

OvchinnikovMMultiDimIntegralsRectangleSEQ::OvchinnikovMMultiDimIntegralsRectangleSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool OvchinnikovMMultiDimIntegralsRectangleSEQ::ValidationImpl() {
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

bool OvchinnikovMMultiDimIntegralsRectangleSEQ::PreProcessingImpl() {
  return true;
}

bool OvchinnikovMMultiDimIntegralsRectangleSEQ::RunImpl() {
  const auto &input = GetInput();
  int n = std::get<0>(input);
  int dim = std::get<1>(input);
  const auto &lower_bounds = std::get<2>(input);
  const auto &upper_bounds = std::get<3>(input);

  std::vector<std::vector<double>> axis_coords(dim);
  double cell_volume = 1.0;

  for (int curr_dim = 0; curr_dim < dim; curr_dim++) {
    double step = (upper_bounds[curr_dim] - lower_bounds[curr_dim]) / n;
    cell_volume *= step;

    axis_coords[curr_dim].resize(n);
    for (int i = 0; i < n; i++) {
      axis_coords[curr_dim][i] = lower_bounds[curr_dim] + ((i + 0.5) * step);
    }
  }

  double integral = 0.0;
  std::vector<double> point(dim);
  std::vector<int> indices(dim, 0);

  bool done = false;
  while (!done) {
    for (int curr_dim = 0; curr_dim < dim; curr_dim++) {
      point[curr_dim] = axis_coords[curr_dim][indices[curr_dim]];
    }

    integral += MultivariableFunction(point);

    int d = dim - 1;
    while (d >= 0) {
      indices[d]++;
      if (indices[d] < n) {
        break;
      }
      indices[d] = 0;
      d--;
    }

    if (d < 0) {
      done = true;
    }
  }

  integral *= cell_volume;
  GetOutput() = integral;
  return true;
}

bool OvchinnikovMMultiDimIntegralsRectangleSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ovchinnikov_m_multidim_integrals_rectangle
