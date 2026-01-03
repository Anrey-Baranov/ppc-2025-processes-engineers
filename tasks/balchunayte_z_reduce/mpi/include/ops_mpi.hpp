#pragma once

#include <vector>

#include "balchunayte_z_reduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace balchunayte_z_reduce {

class BalchunayteZReduceMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit BalchunayteZReduceMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = 0.0;
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int world_rank_{0};
  int world_size_{1};

  int root_{0};

  int local_size_{0};
  std::vector<double> local_data_;
  double local_sum_{0.0};
};

}  // namespace balchunayte_z_reduce
