#pragma once

#include "likhanov_m_elem_vec_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace likhanov_m_elem_vec_sum {

class LikhanovMElemVecSumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LikhanovMElemVecSumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace likhanov_m_elem_vec_sum
