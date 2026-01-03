#pragma once

#include "kopilov_d_ring_2/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kopilov_d_ring_2 {

class KopilovDRingSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KopilovDRingSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kopilov_d_ring_2
