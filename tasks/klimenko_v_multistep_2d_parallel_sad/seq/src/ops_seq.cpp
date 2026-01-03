#include "klimenko_v_multistep_2d_parallel_sad/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "klimenko_v_multistep_2d_parallel_sad/common/include/common.hpp"

namespace klimenko_v_multistep_2d_parallel_sad {

KlimenkoV2DParallelSadSEQ::KlimenkoV2DParallelSadSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool KlimenkoV2DParallelSadSEQ::ValidationImpl() {
  const auto &in = GetInput();

  return in.func != nullptr && in.x_min < in.x_max && in.y_min < in.y_max && in.epsilon > 0.0 && in.r_param > 1.0 &&
         in.max_iterations > 0;
}

bool KlimenkoV2DParallelSadSEQ::PreProcessingImpl() {
  regions_.clear();

  const auto &in = GetInput();

  Region init{};
  init.x_min = in.x_min;
  init.x_max = in.x_max;
  init.y_min = in.y_min;
  init.y_max = in.y_max;

  double xc = 0.5 * (init.x_min + init.x_max);
  double yc = 0.5 * (init.y_min + init.y_max);
  init.f_center = in.func(xc, yc);
  init.characteristic = 0.0;

  regions_.push_back(init);
  return true;
}

bool KlimenkoV2DParallelSadSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  for (int iter = 0; iter < in.max_iterations; ++iter) {
    for (auto &r : regions_) {
      r.characteristic = ComputeCharacteristic(r);
    }

    auto best_it = std::ranges::max_element(
        regions_, [](const Region &a, const Region &b) { return a.characteristic < b.characteristic; });

    Region best = *best_it;

    double dx = best.x_max - best.x_min;
    double dy = best.y_max - best.y_min;

    if (dx < in.epsilon && dy < in.epsilon) {
      out.converged = true;
      out.iterations = iter + 1;
      break;
    }

    regions_.erase(best_it);

    auto [r1, r2] = SplitRegion(best);

    auto eval = [&](Region &r) {
      double xc = 0.5 * (r.x_min + r.x_max);
      double yc = 0.5 * (r.y_min + r.y_max);
      r.f_center = in.func(xc, yc);
    };

    eval(r1);
    eval(r2);

    regions_.push_back(r1);
    regions_.push_back(r2);

    out.iterations = iter + 1;
  }

  return true;
}

bool KlimenkoV2DParallelSadSEQ::PostProcessingImpl() {
  auto &out = GetOutput();

  double best_val = std::numeric_limits<double>::max();

  for (const auto &r : regions_) {
    if (r.f_center < best_val) {
      best_val = r.f_center;
      out.func_min = r.f_center;
      out.x_opt = 0.5 * (r.x_min + r.x_max);
      out.y_opt = 0.5 * (r.y_min + r.y_max);
    }
  }

  return true;
}

double KlimenkoV2DParallelSadSEQ::ComputeCharacteristic(const Region &r) {
  const auto &in = GetInput();

  double dx = r.x_max - r.x_min;
  double dy = r.y_max - r.y_min;

  return -r.f_center + (in.r_param * std::sqrt((dx * dx) + (dy * dy)));
}

std::pair<Region, Region> KlimenkoV2DParallelSadSEQ::SplitRegion(const Region &r) {
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

}  // namespace klimenko_v_multistep_2d_parallel_sad
