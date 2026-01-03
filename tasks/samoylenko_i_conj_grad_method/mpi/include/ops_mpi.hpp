#pragma once

#include "samoylenko_i_conj_grad_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace samoylenko_i_conj_grad_method {

class SamoylenkoIConjGradMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SamoylenkoIConjGradMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace samoylenko_i_conj_grad_method
