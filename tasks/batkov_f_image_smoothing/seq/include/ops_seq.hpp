#pragma once

#include <vector>

#include "batkov_f_image_smoothing/common/include/common.hpp"
#include "task/include/task.hpp"

namespace batkov_f_image_smoothing {

class BatkovFImageSmoothingSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BatkovFImageSmoothingSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<std::vector<float>> gaussian_kernel_;
};

}  // namespace batkov_f_image_smoothing
