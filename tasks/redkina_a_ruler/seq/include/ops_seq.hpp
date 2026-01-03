#pragma once

#include "redkina_a_ruler/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_ruler {

class RedkinaARulerSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RedkinaARulerSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace redkina_a_ruler
