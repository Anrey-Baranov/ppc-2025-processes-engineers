#pragma once

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

class LuchnikovETransmFrAllToOneGatherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LuchnikovETransmFrAllToOneGatherSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
