#pragma once

#include "kopilov_d_shell_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kopilov_d_shell_merge {

class KopilovDShellMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KopilovDShellMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType data_;
};

}  // namespace kopilov_d_shell_merge
