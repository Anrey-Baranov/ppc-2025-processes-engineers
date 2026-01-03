#pragma once

#include "task/include/task.hpp"
#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"

namespace vasiliev_m_bellman_ford_crs {

class VasilievMBellmanFordCrsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VasilievMBellmanFordCrsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_bellman_ford_crs
