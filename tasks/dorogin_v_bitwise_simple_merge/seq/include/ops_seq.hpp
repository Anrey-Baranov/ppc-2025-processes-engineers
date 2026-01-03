#pragma once

#include <vector>

#include "dorogin_v_bitwise_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dorogin_v_bitwise_simple_merge {

class DoroginVBitwiseSimpleMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DoroginVBitwiseSimpleMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> input_data_;
  std::vector<double> output_data_;
};

}  // namespace dorogin_v_bitwise_simple_merge
