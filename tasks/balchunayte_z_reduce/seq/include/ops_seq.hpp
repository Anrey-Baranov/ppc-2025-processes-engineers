#pragma once

#include "balchunayte_z_reduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace balchunayte_z_reduce {

class BalchunayteZReduceSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit BalchunayteZReduceSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace balchunayte_z_reduce
