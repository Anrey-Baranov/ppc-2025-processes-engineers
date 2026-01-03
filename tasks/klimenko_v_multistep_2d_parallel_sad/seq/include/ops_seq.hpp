#pragma once

#include <utility>
#include <vector>

#include "klimenko_v_multistep_2d_parallel_sad/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimenko_v_multistep_2d_parallel_sad {

class KlimenkoV2DParallelSadSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KlimenkoV2DParallelSadSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<Region> regions_;
  [[nodiscard]] static std::pair<Region, Region> SplitRegion(const Region &r);
  double ComputeCharacteristic(const Region &r);
};

}  // namespace klimenko_v_multistep_2d_parallel_sad
