#pragma once

#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_graham_approach {

class RedkinaAGrahamApproachMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() noexcept {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit RedkinaAGrahamApproachMPI(const InType &in);

  static std::vector<Point> GrahamScan(std::vector<Point> points);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<Point> ComputeFinalHull(int rank, std::vector<Point> &all_hull_points);
};

}  // namespace redkina_a_graham_approach
