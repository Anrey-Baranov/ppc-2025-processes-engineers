#pragma once

#include "marin_l_gener_transm_fr_all_to_one_gather/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marin_l_gener_transm_fr_all_to_one_gather {

class MarinLGenerTransmFrAllToOneGatherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MarinLGenerTransmFrAllToOneGatherSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace marin_l_gener_transm_fr_all_to_one_gather
