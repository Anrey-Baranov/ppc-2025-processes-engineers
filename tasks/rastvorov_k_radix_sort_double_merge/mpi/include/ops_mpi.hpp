#pragma once

#include <vector>

#include "rastvorov_k_radix_sort_double_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rastvorov_k_radix_sort_double_merge {

class RastvorovKRadixSortDoubleMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit RastvorovKRadixSortDoubleMergeMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = {};
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> local_;
  std::vector<int> counts_;
  std::vector<int> displs_;
  int world_rank_{0};
  int world_size_{1};
};

}  // namespace rastvorov_k_radix_sort_double_merge
