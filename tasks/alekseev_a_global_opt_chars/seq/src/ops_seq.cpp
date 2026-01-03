#include "alekseev_a_global_opt_chars/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "alekseev_a_global_opt_chars/common/include/common.hpp"

namespace alekseev_a_global_opt_chars {

AlekseevAGlobalOptCharsSEQ::AlekseevAGlobalOptCharsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool AlekseevAGlobalOptCharsSEQ::ValidationImpl() {
  const auto &input = GetInput();

  bool valid = true;
  valid = valid && (input.func != nullptr);
  valid = valid && (input.max_iterations > 0);
  valid = valid && (input.epsilon > 0.0);
  valid = valid && (input.r_param > 1.0);
  valid = valid && (input.x_min < input.x_max);
  valid = valid && (input.y_min < input.y_max);

  return valid;
}

bool AlekseevAGlobalOptCharsSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  trial_points_.clear();
  t_points_.clear();
  lipschitz_estimate_ = 1.0;

  const double t_left = 0.0;
  const double t_right = 1.0;

  t_points_.push_back(t_left);
  t_points_.push_back(t_right);

  const double x0 = PeanoToX(t_left, input.x_min, input.x_max, peano_level_);
  const double y0 = PeanoToY(t_left, input.y_min, input.y_max, peano_level_);
  const double z0 = input.func(x0, y0);

  const double x1 = PeanoToX(t_right, input.x_min, input.x_max, peano_level_);
  const double y1 = PeanoToY(t_right, input.y_min, input.y_max, peano_level_);
  const double z1 = input.func(x1, y1);

  trial_points_.emplace_back(x0, y0, z0);
  trial_points_.emplace_back(x1, y1, z1);

  return true;
}

bool AlekseevAGlobalOptCharsSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  if (t_points_.size() < 2 || trial_points_.size() < 2) {
    if (!PreProcessingImpl()) {
      return false;
    }
  }

  output.iterations = 0;
  output.converged = false;

  for (int iter = 0; iter < input.max_iterations; ++iter) {
    SortingTrials();

    lipschitz_estimate_ = ComputeLipschitzEstimate();
    if (lipschitz_estimate_ < 1e-10) {
      lipschitz_estimate_ = 1.0;
    }

    const int best_interval_idx = BestInterval();

    const double t_left = t_points_[best_interval_idx];
    const double t_right = t_points_[best_interval_idx + 1];
    const double z_left = trial_points_[best_interval_idx].z;
    const double z_right = trial_points_[best_interval_idx + 1].z;

    const double interval_len = t_right - t_left;
    if (interval_len < input.epsilon) {
      output.converged = true;
      output.iterations = iter + 1;
      break;
    }

    const double m_val = input.r_param * lipschitz_estimate_;

    double t_new = (0.5 * (t_left + t_right)) - ((z_right - z_left) / (2.0 * m_val));

    t_new = std::max(t_left + 1e-12, std::min(t_new, t_right - 1e-12));

    const double z_new = PerformTrial(t_new);

    const double x_new = PeanoToX(t_new, input.x_min, input.x_max, peano_level_);
    const double y_new = PeanoToY(t_new, input.y_min, input.y_max, peano_level_);

    t_points_.push_back(t_new);
    trial_points_.emplace_back(x_new, y_new, z_new);

    output.iterations = iter + 1;
  }

  return true;
}

bool AlekseevAGlobalOptCharsSEQ::PostProcessingImpl() {
  auto &output = GetOutput();

  double best_z = std::numeric_limits<double>::max();
  std::size_t best_idx = 0;

  for (std::size_t i = 0; i < trial_points_.size(); ++i) {
    if (trial_points_[i].z < best_z) {
      best_z = trial_points_[i].z;
      best_idx = i;
    }
  }
  output.x_opt = trial_points_[best_idx].x;
  output.y_opt = trial_points_[best_idx].y;
  output.func_min = best_z;

  return true;
}

void AlekseevAGlobalOptCharsSEQ::SortingTrials() {
  std::vector<std::size_t> idx(t_points_.size());
  for (std::size_t i = 0; i < idx.size(); ++i) {
    idx[i] = i;
  }

  std::ranges::sort(idx, [this](std::size_t a, std::size_t b) { return t_points_[a] < t_points_[b]; });

  std::vector<double> sorted_t(t_points_.size());
  std::vector<TrialPoint> sorted_trials(trial_points_.size());

  for (std::size_t i = 0; i < idx.size(); ++i) {
    sorted_t[i] = t_points_[idx[i]];
    sorted_trials[i] = trial_points_[idx[i]];
  }

  t_points_ = std::move(sorted_t);
  trial_points_ = std::move(sorted_trials);
}

double AlekseevAGlobalOptCharsSEQ::ComputeLipschitzEstimate() {
  double max_slope = 0.0;

  for (std::size_t i = 1; i < t_points_.size(); ++i) {
    const double dt = t_points_[i] - t_points_[i - 1];
    if (dt > 1e-15) {
      const double dz = std::abs(trial_points_[i].z - trial_points_[i - 1].z);
      max_slope = std::max(max_slope, dz / dt);
    }
  }

  return (max_slope > 0.0) ? max_slope : 1.0;
}

double AlekseevAGlobalOptCharsSEQ::ComputeIntervalCharacteristic(int interval_idx, double m_val) const {
  const double t_i = t_points_[interval_idx];
  const double t_j = t_points_[interval_idx + 1];
  const double z_i = trial_points_[interval_idx].z;
  const double z_j = trial_points_[interval_idx + 1].z;

  const double dt = t_j - t_i;
  const double dz = z_j - z_i;

  return (m_val * dt) + ((dz * dz) / (m_val * dt)) - (2.0 * (z_j + z_i));
}

int AlekseevAGlobalOptCharsSEQ::BestInterval() {
  const auto &input = GetInput();
  const double m_val = input.r_param * lipschitz_estimate_;

  double best_char = -std::numeric_limits<double>::max();
  int best_idx = 0;

  for (std::size_t i = 0; i + 1 < t_points_.size(); ++i) {
    const double c = ComputeIntervalCharacteristic(static_cast<int>(i), m_val);
    if (c > best_char) {
      best_char = c;
      best_idx = static_cast<int>(i);
    }
  }

  return best_idx;
}

double AlekseevAGlobalOptCharsSEQ::PerformTrial(double t) {
  const auto &input = GetInput();

  const double x = PeanoToX(t, input.x_min, input.x_max, peano_level_);
  const double y = PeanoToY(t, input.y_min, input.y_max, peano_level_);

  return input.func(x, y);
}

}  // namespace alekseev_a_global_opt_chars
