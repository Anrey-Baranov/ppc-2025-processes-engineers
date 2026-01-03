#pragma once

#include "sinev_a_allreduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sinev_a_allreduce {

class SinevAAllreduceSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SinevAAllreduceSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sinev_a_allreduce
