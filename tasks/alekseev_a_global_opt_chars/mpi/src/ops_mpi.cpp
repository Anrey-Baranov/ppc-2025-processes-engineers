#include "alekseev_a_global_opt_chars/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "alekseev_a_global_opt_chars/common/include/common.hpp"

namespace alekseev_a_global_opt_chars {

namespace {

void ComputeDistribution(int num_items, int world_size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.assign(static_cast<std::size_t>(world_size), 0);
  displs.assign(static_cast<std::size_t>(world_size), 0);

  const int base = num_items / world_size;
  const int rem = num_items % world_size;

  for (int i = 0; i < world_size; ++i) {
    counts[static_cast<std::size_t>(i)] = base + ((i < rem) ? 1 : 0);
    displs[static_cast<std::size_t>(i)] =
        (i == 0) ? 0 : (displs[static_cast<std::size_t>(i - 1)] + counts[static_cast<std::size_t>(i - 1)]);
  }
}

void PackIntervalsData(const std::vector<double> &t_points, const std::vector<TrialPoint> &trial_points,
                       int interval_count, std::vector<double> &packed) {
  packed.resize(static_cast<std::size_t>(interval_count) * 4);

  for (int i = 0; i < interval_count; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    packed[(idx * 4) + 0] = t_points[idx];
    packed[(idx * 4) + 1] = t_points[idx + 1];
    packed[(idx * 4) + 2] = trial_points[idx].z;
    packed[(idx * 4) + 3] = trial_points[idx + 1].z;
  }
}

void ComputeLocalCharacteristics(const std::vector<double> &local_interval_data, int local_interval_count, double m_val,
                                 std::vector<double> &local_chars) {
  local_chars.resize(static_cast<std::size_t>(local_interval_count));

  for (int i = 0; i < local_interval_count; ++i) {
    const auto idx = static_cast<std::size_t>(i);

    const double t_left = local_interval_data[(idx * 4) + 0];
    const double t_right = local_interval_data[(idx * 4) + 1];
    const double z_left = local_interval_data[(idx * 4) + 2];
    const double z_right = local_interval_data[(idx * 4) + 3];

    const double dt = t_right - t_left;
    const double dz = z_right - z_left;

    local_chars[idx] = (m_val * dt) + ((dz * dz) / (m_val * dt)) - (2.0 * (z_right + z_left));
  }
}

void GatherCharacteristicsToRoot(const std::vector<double> &local_chars, const std::vector<int> &counts,
                                 const std::vector<int> &displs, int world_rank, int world_size,
                                 std::vector<double> &all_chars) {
  const int local_count = counts[static_cast<std::size_t>(world_rank)];

  if (world_rank == 0) {
    for (int i = 0; i < counts[0]; ++i) {
      all_chars[static_cast<std::size_t>(i)] = local_chars[static_cast<std::size_t>(i)];
    }

    for (int proc = 1; proc < world_size; ++proc) {
      const int proc_count = counts[static_cast<std::size_t>(proc)];
      if (proc_count <= 0) {
        continue;
      }

      std::vector<double> recv(proc_count);
      MPI_Recv(recv.data(), proc_count, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      const int disp = displs[static_cast<std::size_t>(proc)];
      for (int i = 0; i < proc_count; ++i) {
        all_chars[static_cast<std::size_t>(disp) + i] = recv[static_cast<std::size_t>(i)];
      }
    }
  } else {
    if (local_count > 0) {
      MPI_Send(local_chars.data(), local_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }
}

}  // namespace

AlekseevAGlobalOptCharsMPI::AlekseevAGlobalOptCharsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
}

bool AlekseevAGlobalOptCharsMPI::ValidationImpl() {
  if (world_rank_ != 0) {
    return true;
  }

  const auto &input = GetInput();

  bool valid = true;
  valid = valid && (input.func != nullptr);
  valid = valid && (input.x_min < input.x_max);
  valid = valid && (input.y_min < input.y_max);
  valid = valid && (input.epsilon > 0.0);
  valid = valid && (input.r_param > 1.0);
  valid = valid && (input.max_iterations > 0);

  return valid;
}

bool AlekseevAGlobalOptCharsMPI::PreProcessingImpl() {
  trial_points_.clear();
  t_points_.clear();
  lipschitz_estimate_ = 1.0;
  return true;
}

bool AlekseevAGlobalOptCharsMPI::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  trial_points_.clear();
  t_points_.clear();

  std::vector<double> init_pack(8, 0.0);

  if (world_rank_ == 0) {
    const double t0 = 0.0;
    const double t1 = 1.0;

    const double x0 = PeanoToX(t0, input.x_min, input.x_max, peano_level_);
    const double y0 = PeanoToY(t0, input.y_min, input.y_max, peano_level_);
    const double z0 = input.func(x0, y0);

    const double x1 = PeanoToX(t1, input.x_min, input.x_max, peano_level_);
    const double y1 = PeanoToY(t1, input.y_min, input.y_max, peano_level_);
    const double z1 = input.func(x1, y1);

    init_pack[0] = t0;
    init_pack[1] = x0;
    init_pack[2] = y0;
    init_pack[3] = z0;
    init_pack[4] = t1;
    init_pack[5] = x1;
    init_pack[6] = y1;
    init_pack[7] = z1;
  }

  MPI_Bcast(init_pack.data(), 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  t_points_.push_back(init_pack[0]);
  t_points_.push_back(init_pack[4]);
  trial_points_.emplace_back(init_pack[1], init_pack[2], init_pack[3]);
  trial_points_.emplace_back(init_pack[5], init_pack[6], init_pack[7]);

  output.iterations = 0;
  output.converged = false;

  for (int iter = 0; iter < input.max_iterations; ++iter) {
    BroadcastTrialData();
    SortingTrials();

    lipschitz_estimate_ = ComputeLipschitzEstimate();
    if (lipschitz_estimate_ < 1e-10) {
      lipschitz_estimate_ = 1.0;
    }

    const double m_val = input.r_param * lipschitz_estimate_;

    std::vector<double> characteristics;
    CharacteristicsParallelComputed(m_val, characteristics);

    const int best_interval_idx = BestInterval(characteristics);

    const double t_left = t_points_[best_interval_idx];
    const double t_right = t_points_[best_interval_idx + 1];
    const double z_left = trial_points_[best_interval_idx].z;
    const double z_right = trial_points_[best_interval_idx + 1].z;

    const double interval_len = t_right - t_left;

    int stop_flag = (interval_len < input.epsilon) ? 1 : 0;
    MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (stop_flag != 0) {
      output.converged = true;
      output.iterations = iter + 1;
      break;
    }

    double t_new = (0.5 * (t_left + t_right)) - ((z_right - z_left) / (2.0 * m_val));
    t_new = std::max(t_left + 1e-12, std::min(t_new, t_right - 1e-12));

    if (world_rank_ == 0) {
      const double z_new = PerformTrial(t_new);
      const double x_new = PeanoToX(t_new, input.x_min, input.x_max, peano_level_);
      const double y_new = PeanoToY(t_new, input.y_min, input.y_max, peano_level_);

      t_points_.push_back(t_new);
      trial_points_.emplace_back(x_new, y_new, z_new);
    }

    output.iterations = iter + 1;
  }

  BroadcastTrialData();

  return true;
}

bool AlekseevAGlobalOptCharsMPI::PostProcessingImpl() {
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

void AlekseevAGlobalOptCharsMPI::SortingTrials() {
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

double AlekseevAGlobalOptCharsMPI::ComputeLipschitzEstimate() {
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

void AlekseevAGlobalOptCharsMPI::CharacteristicsParallelComputed(double m_val, std::vector<double> &characteristics) {
  const int interval_count = static_cast<int>(t_points_.size()) - 1;
  if (interval_count <= 0) {
    characteristics.clear();
    return;
  }

  std::vector<int> counts;
  std::vector<int> displs;
  ComputeDistribution(interval_count, world_size_, counts, displs);

  std::vector<double> packed_intervals;
  if (world_rank_ == 0) {
    PackIntervalsData(t_points_, trial_points_, interval_count, packed_intervals);
  } else {
    packed_intervals.resize(static_cast<std::size_t>(interval_count) * 4);
  }

  std::vector<int> send_counts(static_cast<std::size_t>(world_size_));
  std::vector<int> send_displs(static_cast<std::size_t>(world_size_));
  for (int i = 0; i < world_size_; ++i) {
    send_counts[static_cast<std::size_t>(i)] = counts[static_cast<std::size_t>(i)] * 4;
    send_displs[static_cast<std::size_t>(i)] = displs[static_cast<std::size_t>(i)] * 4;
  }

  const int local_interval_count = counts[static_cast<std::size_t>(world_rank_)];
  std::vector<double> local_interval_data(static_cast<std::size_t>(local_interval_count) * 4);

  MPI_Scatterv(packed_intervals.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE, local_interval_data.data(),
               local_interval_count * 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_chars;
  ComputeLocalCharacteristics(local_interval_data, local_interval_count, m_val, local_chars);

  characteristics.resize(static_cast<std::size_t>(interval_count));
  GatherCharacteristicsToRoot(local_chars, counts, displs, world_rank_, world_size_, characteristics);

  MPI_Bcast(characteristics.data(), interval_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int AlekseevAGlobalOptCharsMPI::BestInterval(const std::vector<double> &characteristics) {
  double best_char = -std::numeric_limits<double>::max();
  int best_idx = 0;

  for (std::size_t i = 0; i < characteristics.size(); ++i) {
    if (characteristics[i] > best_char) {
      best_char = characteristics[i];
      best_idx = static_cast<int>(i);
    }
  }

  return best_idx;
}

double AlekseevAGlobalOptCharsMPI::PerformTrial(double t) {
  const auto &input = GetInput();

  const double x = PeanoToX(t, input.x_min, input.x_max, peano_level_);
  const double y = PeanoToY(t, input.y_min, input.y_max, peano_level_);

  return input.func(x, y);
}

void AlekseevAGlobalOptCharsMPI::BroadcastTrialData() {
  int n = static_cast<int>(t_points_.size());
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank_ != 0) {
    t_points_.resize(static_cast<std::size_t>(n));
    trial_points_.resize(static_cast<std::size_t>(n));
  }

  MPI_Bcast(t_points_.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> packed_trials(static_cast<std::size_t>(n) * 3);

  if (world_rank_ == 0) {
    for (int i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      packed_trials[(idx * 3) + 0] = trial_points_[idx].x;
      packed_trials[(idx * 3) + 1] = trial_points_[idx].y;
      packed_trials[(idx * 3) + 2] = trial_points_[idx].z;
    }
  }

  MPI_Bcast(packed_trials.data(), n * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank_ != 0) {
    for (int i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      trial_points_[idx].x = packed_trials[(idx * 3) + 0];
      trial_points_[idx].y = packed_trials[(idx * 3) + 1];
      trial_points_[idx].z = packed_trials[(idx * 3) + 2];
    }
  }
}

}  // namespace alekseev_a_global_opt_chars
