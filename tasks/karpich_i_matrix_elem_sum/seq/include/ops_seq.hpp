#pragma once

#include "karpich_i_matrix_elem_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace karpich_i_matrix_elem_sum {

class KarpichIMatrixElemSumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KarpichIMatrixElemSumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace karpich_i_matrix_elem_sum
