#pragma once

#include "bortsova_a_transmission_gather/common/include/common.hpp"
#include "task/include/task.hpp"

namespace bortsova_a_transmission_gather {

class BortsovaATransmissionGatherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BortsovaATransmissionGatherSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace bortsova_a_transmission_gather
