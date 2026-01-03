#pragma once

#include <cstdint>
#include <vector>

#include "dorogin_v_contrast_enhancement/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dorogin_v_contrast_enhancement {

class DoroginVContrastEnhancementMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DoroginVContrastEnhancementMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<uint8_t> image_;
  std::vector<uint8_t> local_image_;
  std::vector<uint8_t> result_;
  int world_rank_{0};
  int world_size_{1};
  int local_size_{0};
};

}  // namespace dorogin_v_contrast_enhancement
