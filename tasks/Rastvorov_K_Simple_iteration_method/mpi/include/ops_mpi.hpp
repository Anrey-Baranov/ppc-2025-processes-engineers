#pragma once

#include "Rastvorov_K_Simple_iteration_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rastvorov_k_simple_iteration_method {

class RastvorovKSimpleIterationMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RastvorovKSimpleIterationMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace rastvorov_k_simple_iteration_method
