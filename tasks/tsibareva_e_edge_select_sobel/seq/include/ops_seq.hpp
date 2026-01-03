#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "tsibareva_e_edge_select_sobel/common/include/common.hpp"

namespace tsibareva_e_edge_select_sobel {

class TsibarevaEEdgeSelectSobelSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TsibarevaEEdgeSelectSobelSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int GradientX(int x, int y);
  int GradientY(int x, int y);

  int height_ = 0;
  int width_ = 0;
  int threshold_ = 0;

  std::vector<int> input_pixels_;
};

}  // namespace tsibareva_e_edge_select_sobel
