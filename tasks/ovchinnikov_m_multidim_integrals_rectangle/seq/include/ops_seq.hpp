#pragma once

#include "ovchinnikov_m_multidim_integrals_rectangle/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ovchinnikov_m_multidim_integrals_rectangle {

class OvchinnikovMMultiDimIntegralsRectangleSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OvchinnikovMMultiDimIntegralsRectangleSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ovchinnikov_m_multidim_integrals_rectangle
