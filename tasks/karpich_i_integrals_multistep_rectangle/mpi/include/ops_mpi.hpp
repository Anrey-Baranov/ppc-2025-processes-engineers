#pragma once

#include "karpich_i_integrals_multistep_rectangle/common/include/common.hpp"
#include "task/include/task.hpp"

namespace karpich_i_integrals_multistep_rectangle {

class KarpichIIntegralsMultistepRectangleMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KarpichIIntegralsMultistepRectangleMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace karpich_i_integrals_multistep_rectangle
