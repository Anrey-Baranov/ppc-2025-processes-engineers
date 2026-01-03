#pragma once

#include <cstdint>
#include <vector>

#include "shilin_n_gauss_filter_vertical_split/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shilin_n_gauss_filter_vertical_split {

class ShilinNGaussFilterVerticalSplitSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShilinNGaussFilterVerticalSplitSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ApplyGaussianKernel(const std::vector<uint8_t> &input, std::vector<uint8_t> &output, int width,
                                  int height, int channels);
  static double GetPixelValue(const std::vector<uint8_t> &pixels, int x, int y, int width, int height, int channels,
                              int channel);
};

}  // namespace shilin_n_gauss_filter_vertical_split
