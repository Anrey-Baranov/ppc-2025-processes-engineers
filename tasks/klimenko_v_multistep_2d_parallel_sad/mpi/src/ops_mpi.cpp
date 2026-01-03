#include "klimenko_v_multistep_2d_parallel_sad/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "klimenko_v_multistep_2d_parallel_sad/common/include/common.hpp"

namespace klimenko_v_multistep_2d_parallel_sad {

KlimenkoV2DParallelSadMPI::KlimenkoV2DParallelSadMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
}

bool KlimenkoV2DParallelSadMPI::ValidationImpl() {
  if (world_rank_ != 0) {
    return true;
  }

  const auto &in = GetInput();
  return in.func != nullptr && in.x_min < in.x_max && in.y_min < in.y_max && in.epsilon > 0.0 && in.r_param > 1.0 &&
         in.max_iterations > 0;
}

bool KlimenkoV2DParallelSadMPI::PreProcessingImpl() {
  regions_.clear();

  if (world_rank_ == 0) {
    const auto &in = GetInput();

    Region r{};
    r.x_min = in.x_min;
    r.x_max = in.x_max;
    r.y_min = in.y_min;
    r.y_max = in.y_max;
    double xc = 0.5 * (r.x_min + r.x_max);
    double yc = 0.5 * (r.y_min + r.y_max);
    r.f_center = in.func(xc, yc);
    r.characteristic = 0.0;

    regions_.push_back(r);
    epsilon_ = in.epsilon;
  }

  MPI_Bcast(&epsilon_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

double KlimenkoV2DParallelSadMPI::ComputeCharacteristic(const Region &r) {
  const auto &in = GetInput();

  double dx = r.x_max - r.x_min;
  double dy = r.y_max - r.y_min;

  return (-r.f_center) + (in.r_param * std::sqrt((dx * dx) + (dy * dy)));
  ;
}

std::pair<Region, Region> KlimenkoV2DParallelSadMPI::SplitRegion(const Region &r) {
  Region r1 = r;
  Region r2 = r;

  if ((r.x_max - r.x_min) >= (r.y_max - r.y_min)) {
    double xm = 0.5 * (r.x_min + r.x_max);
    r1.x_max = xm;
    r2.x_min = xm;
  } else {
    double ym = 0.5 * (r.y_min + r.y_max);
    r1.y_max = ym;
    r2.y_min = ym;
  }

  return {r1, r2};
}

bool KlimenkoV2DParallelSadMPI::RunImpl() {
  const auto &in = GetInput();

  for (int iter = 0; iter < in.max_iterations; ++iter) {
    const Region local_best = FindLocalBestRegion();
    const double local_char = ComputeCharacteristic(local_best);

    const Region best_region = FindGlobalBestRegion(local_best, local_char);

    SplitBestRegion(best_region);

    if (CheckStopCondition()) {
      break;
    }
  }
  return true;
}

bool KlimenkoV2DParallelSadMPI::PostProcessingImpl() {
  auto &out = GetOutput();

  double func_min = 0.0;
  double x_opt = 0.0;
  double y_opt = 0.0;

  if (world_rank_ == 0) {
    double best_val = std::numeric_limits<double>::max();

    for (const auto &r : regions_) {
      if (r.f_center < best_val) {
        best_val = r.f_center;
        func_min = r.f_center;
        x_opt = 0.5 * (r.x_min + r.x_max);
        y_opt = 0.5 * (r.y_min + r.y_max);
      }
    }
  }

  MPI_Bcast(&func_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&x_opt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&y_opt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  out.func_min = func_min;
  out.x_opt = x_opt;
  out.y_opt = y_opt;

  return true;
}

Region KlimenkoV2DParallelSadMPI::FindLocalBestRegion() {
  Region best{};
  double best_char = -std::numeric_limits<double>::infinity();

  for (auto i = static_cast<std::size_t>(world_rank_); i < regions_.size();
       i += static_cast<std::size_t>(world_size_)) {
    const double ch = ComputeCharacteristic(regions_[i]);
    if (ch > best_char) {
      best_char = ch;
      best = regions_[i];
    }
  }
  return best;
}

Region KlimenkoV2DParallelSadMPI::FindGlobalBestRegion(const Region &local_best, double local_char) const {
  struct {
    double value;
    int rank;
  } local_pair{.value = local_char, .rank = world_rank_}, global_pair{};

  MPI_Reduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

  Region best{};
  if (world_rank_ == global_pair.rank) {
    best = local_best;
  }

  MPI_Bcast(&best, static_cast<int>(sizeof(Region)), MPI_BYTE, global_pair.rank, MPI_COMM_WORLD);
  return best;
}

void KlimenkoV2DParallelSadMPI::SplitBestRegion(const Region &best_region) {
  if (world_rank_ != 0) {
    return;
  }

  const auto split = SplitRegion(best_region);

  std::erase_if(regions_, [&](const Region &r) {
    return r.x_min == best_region.x_min && r.x_max == best_region.x_max && r.y_min == best_region.y_min &&
           r.y_max == best_region.y_max;
  });

  auto eval = [&](Region &r) {
    const double xc = 0.5 * (r.x_min + r.x_max);
    const double yc = 0.5 * (r.y_min + r.y_max);
    r.f_center = GetInput().func(xc, yc);
  };

  Region r1 = split.first;
  Region r2 = split.second;

  eval(r1);
  eval(r2);

  regions_.push_back(r1);
  regions_.push_back(r2);
}

bool KlimenkoV2DParallelSadMPI::CheckStopCondition() const {
  int stop_flag = 0;

  if (world_rank_ == 0 && !regions_.empty()) {
    const auto &last = regions_.back();
    const double dx = last.x_max - last.x_min;
    const double dy = last.y_max - last.y_min;
    stop_flag = (dx < epsilon_ && dy < epsilon_) ? 1 : 0;
  }

  MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return stop_flag == 1;
}

}  // namespace klimenko_v_multistep_2d_parallel_sad
