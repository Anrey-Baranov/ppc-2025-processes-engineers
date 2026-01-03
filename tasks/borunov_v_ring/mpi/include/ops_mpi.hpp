#pragma once

#include "borunov_v_ring/common/include/common.hpp"
#include "task/include/task.hpp"

namespace borunov_v_ring {

class BorunovVRingMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BorunovVRingMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace borunov_v_ring
