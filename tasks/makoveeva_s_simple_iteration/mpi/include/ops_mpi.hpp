#ifndef TASKS_MAKOVEEVA_S_SIMPLE_ITERATION_MPI_INCLUDE_OPS_MPI_HPP_
#define TASKS_MAKOVEEVA_S_SIMPLE_ITERATION_MPI_INCLUDE_OPS_MPI_HPP_

#include "makoveeva_s_simple_iteration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makoveeva_s_simple_iteration {

class MakoveevaSSimpleIterationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit MakoveevaSSimpleIterationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace makoveeva_s_simple_iteration

#endif  // TASKS_MAKOVEEVA_S_SIMPLE_ITERATION_MPI_INCLUDE_OPS_MPI_HPP_
