#pragma once

#include "borunov_v_block_partitioning/common/include/common.hpp"
#include "task/include/task.hpp"

namespace borunov_v_block_partitioning {

class BorunovVBlockPartitioningMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BorunovVBlockPartitioningMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace borunov_v_block_partitioning
