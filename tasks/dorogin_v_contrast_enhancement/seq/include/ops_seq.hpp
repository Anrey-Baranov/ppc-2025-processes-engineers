#pragma once

#include <cstdint>
#include <vector>

#include "dorogin_v_contrast_enhancement/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dorogin_v_contrast_enhancement {

class DoroginVContrastEnhancementSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DoroginVContrastEnhancementSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<uint8_t> image_;
  std::vector<uint8_t> result_;
};

}  // namespace dorogin_v_contrast_enhancement
