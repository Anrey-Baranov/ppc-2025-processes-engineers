#pragma once

#include <vector>

#include "kopilov_d_shell_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kopilov_d_shell_merge {

class KopilovDShellMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KopilovDShellMergeMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> local_;
  std::vector<int> counts_;
  std::vector<int> displs_;
  int world_rank_{0};
  int world_size_{1};
};

}  // namespace kopilov_d_shell_merge
