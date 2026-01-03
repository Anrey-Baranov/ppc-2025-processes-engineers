#pragma once

#include <cstdint>
#include <vector>

#include "marin_l_linear_filter_vertical/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marin_l_linear_filter_vertical {

class MarinLLinearFilterVerticalSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MarinLLinearFilterVerticalSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<uint8_t> input_pixels_;
  std::vector<uint8_t> output_pixels_;
  int width_ = 0;
  int height_ = 0;
};

}  // namespace marin_l_linear_filter_vertical
