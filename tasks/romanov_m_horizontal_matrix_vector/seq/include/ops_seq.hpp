#pragma once

#include "romanov_m_horizontal_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanov_m_horizontal_matrix_vector {

class RomanovMHorizontalMatrixVectorSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit RomanovMHorizontalMatrixVectorSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace romanov_m_horizontal_matrix_vector
