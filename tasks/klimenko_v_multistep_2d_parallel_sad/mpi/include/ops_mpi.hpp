#pragma once

#include <utility>
#include <vector>

#include "klimenko_v_multistep_2d_parallel_sad/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimenko_v_multistep_2d_parallel_sad {

class KlimenkoV2DParallelSadMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KlimenkoV2DParallelSadMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] static std::pair<Region, Region> SplitRegion(const Region &r);
  double ComputeCharacteristic(const Region &r);
  Region FindLocalBestRegion();
  [[nodiscard]] Region FindGlobalBestRegion(const Region &local_best, double local_char) const;
  [[nodiscard]] bool CheckStopCondition() const;
  void SplitBestRegion(const Region &best_region);

  int world_rank_{0};
  int world_size_{0};
  std::vector<Region> regions_;
  double epsilon_{};
};

}  // namespace klimenko_v_multistep_2d_parallel_sad
