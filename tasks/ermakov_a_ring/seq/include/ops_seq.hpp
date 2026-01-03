#pragma once

#include "ermakov_a_ring/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ermakov_a_ring {

class ErmakovATestTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ErmakovATestTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ermakov_a_ring
