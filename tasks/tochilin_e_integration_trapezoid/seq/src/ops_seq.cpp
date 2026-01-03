#include "tochilin_e_integration_trapezoid/seq/include/ops_seq.hpp"

#include <cmath>

#include "tochilin_e_integration_trapezoid/common/include/common.hpp"

namespace tochilin_e_integration_trapezoid {

TochilinEIntegrationTrapezoidSEQ::TochilinEIntegrationTrapezoidSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TochilinEIntegrationTrapezoidSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.num_intervals > 0 && input.lower_bound < input.upper_bound && input.function != nullptr;
}

bool TochilinEIntegrationTrapezoidSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  lower_bound_ = input.lower_bound;
  upper_bound_ = input.upper_bound;
  num_intervals_ = input.num_intervals;
  function_ = input.function;
  result_ = 0.0;
  return true;
}

bool TochilinEIntegrationTrapezoidSEQ::RunImpl() {
  double step = (upper_bound_ - lower_bound_) / num_intervals_;

  double sum = (function_(lower_bound_) + function_(upper_bound_)) / 2.0;

  for (int i = 1; i < num_intervals_; ++i) {
    double x = lower_bound_ + (i * step);
    sum += function_(x);
  }

  result_ = step * sum;
  return true;
}

bool TochilinEIntegrationTrapezoidSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace tochilin_e_integration_trapezoid
