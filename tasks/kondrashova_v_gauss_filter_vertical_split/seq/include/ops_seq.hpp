#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

class KondrashovaVGaussFilterVerticalSplitSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KondrashovaVGaussFilterVerticalSplitSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static const std::array<std::array<int, 3>, 3> kGaussKernel;
  static const int kGaussKernelSum;

  [[nodiscard]] static uint8_t ApplyGaussToPixel(const std::vector<uint8_t> &pixels, int width, int height,
                                                 int channels, int px, int py, int channel);
};

}  // namespace kondrashova_v_gauss_filter_vertical_split
