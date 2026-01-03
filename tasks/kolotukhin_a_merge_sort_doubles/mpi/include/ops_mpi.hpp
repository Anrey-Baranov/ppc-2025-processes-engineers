#pragma once

#include "kolotukhin_a_merge_sort_doubles/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kolotukhin_a_merge_sort_doubles {

class KolotukhinAMergeSortDoublesMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KolotukhinAMergeSortDoublesMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kolotukhin_a_merge_sort_doubles
