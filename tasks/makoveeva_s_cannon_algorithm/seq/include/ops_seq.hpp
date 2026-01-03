#pragma once

#include "makoveeva_s_cannon_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makoveeva_s_cannon_algorithm {

class MakoveevaSCannonAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MakoveevaSCannonAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace makoveeva_s_cannon_algorithm
