#pragma once

#include "kolotukhin_a_hypercube/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kolotukhin_a_hypercube {

class KolotukhinAHypercubeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KolotukhinAHypercubeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kolotukhin_a_hypercube
