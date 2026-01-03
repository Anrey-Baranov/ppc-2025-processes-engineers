#include "tochilin_e_integral_trapezium/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "tochilin_e_integral_trapezium/common/include/common.hpp"

namespace tochilin_e_integral_trapezium {

TochilinEIntegralTrapeziumSEQ::TochilinEIntegralTrapeziumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TochilinEIntegralTrapeziumSEQ::ValidationImpl() {
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

  return true;
}

bool TochilinEIntegralTrapeziumSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  lower_bounds_ = input.lower_bounds;
  upper_bounds_ = input.upper_bounds;
  num_steps_ = input.num_steps;
  func_ = input.func;
  result_ = 0.0;
  return true;
}

double TochilinEIntegralTrapeziumSEQ::ComputeIntegral() {
  std::size_t dimensions = lower_bounds_.size();
  std::vector<double> step_sizes(dimensions);
  for (std::size_t idx = 0; idx < dimensions; ++idx) {
    step_sizes[idx] = (upper_bounds_[idx] - lower_bounds_[idx]) / num_steps_;
  }

  int total_points = 1;
  for (std::size_t idx = 0; idx < dimensions; ++idx) {
    total_points *= (num_steps_ + 1);
  }

  double sum = 0.0;
  std::vector<double> point(dimensions);

  for (int idx = 0; idx < total_points; ++idx) {
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

bool TochilinEIntegralTrapeziumSEQ::RunImpl() {
  result_ = ComputeIntegral();
  return true;
}

bool TochilinEIntegralTrapeziumSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace tochilin_e_integral_trapezium
