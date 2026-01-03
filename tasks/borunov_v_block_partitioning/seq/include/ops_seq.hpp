#pragma once

#include "borunov_v_block_partitioning/common/include/common.hpp"
#include "task/include/task.hpp"

namespace borunov_v_block_partitioning {

class BorunovVBlockPartitioningSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BorunovVBlockPartitioningSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace borunov_v_block_partitioning
