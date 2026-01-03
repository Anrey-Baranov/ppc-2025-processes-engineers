#pragma once
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shemetov_d_gauss_filter_linear {

class GaussFilterSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit GaussFilterSEQ(const InType &in);
  static Pixel ApplyKernel(const InType &in, int i, int j, const std::vector<std::vector<float>> &kernel);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  inline static int width = 0;
  inline static int height = 0;
};

}  // namespace shemetov_d_gauss_filter_linear
