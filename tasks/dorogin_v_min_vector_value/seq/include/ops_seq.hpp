#pragma once

#include "dorogin_v_min_vector_value/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dorogin_v_min_vector_value {

class DoroginVMinVectorValueSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DoroginVMinVectorValueSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dorogin_v_min_vector_value
