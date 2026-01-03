#pragma once

#include "task/include/task.hpp"
#include "vasiliev_m_bubble_sort/common/include/common.hpp"

namespace vasiliev_m_bubble_sort {

class VasilievMBubbleSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VasilievMBubbleSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_bubble_sort
