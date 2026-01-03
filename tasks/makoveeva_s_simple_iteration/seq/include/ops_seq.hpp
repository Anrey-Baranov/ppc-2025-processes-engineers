#ifndef TASKS_MAKOVEEVA_S_SIMPLE_ITERATION_SEQ_INCLUDE_OPS_SEQ_HPP_
#define TASKS_MAKOVEEVA_S_SIMPLE_ITERATION_SEQ_INCLUDE_OPS_SEQ_HPP_

#include "makoveeva_s_simple_iteration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makoveeva_s_simple_iteration {

class MakoveevaSSimpleIterationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit MakoveevaSSimpleIterationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace makoveeva_s_simple_iteration

#endif  // TASKS_MAKOVEEVA_S_SIMPLE_ITERATION_SEQ_INCLUDE_OPS_SEQ_HPP_
