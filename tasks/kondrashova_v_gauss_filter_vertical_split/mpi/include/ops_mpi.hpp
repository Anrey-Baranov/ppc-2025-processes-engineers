#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

class KondrashovaVGaussFilterVerticalSplitMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KondrashovaVGaussFilterVerticalSplitMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static const std::array<std::array<int, 3>, 3> kGaussKernel;
  static const int kGaussKernelSum;

  void BroadcastImageDimensions(int &width, int &height, int &channels);

  static void CalculateColumnDistribution(int width, int size, std::vector<int> &col_counts,
                                          std::vector<int> &col_offsets);

  void DistributeImageData(int rank, int size, int width, int height, int channels, const std::vector<int> &col_counts,
                           const std::vector<int> &col_offsets, std::vector<uint8_t> &local_data, int extended_cols);

  static void ApplyGaussFilterToLocalData(const std::vector<uint8_t> &local_data, std::vector<uint8_t> &local_result,
                                          int extended_cols, int local_cols, int height, int channels,
                                          int offset_in_extended);

  void GatherResults(int rank, int size, int width, int height, int channels, const std::vector<int> &col_counts,
                     const std::vector<int> &col_offsets, int local_start_col, int local_cols,
                     const std::vector<uint8_t> &local_result);

  void BroadcastResultToAllProcesses(int width, int height, int channels);

  [[nodiscard]] static uint8_t ApplyGaussToLocalPixel(const std::vector<uint8_t> &local_data, int local_width,
                                                      int height, int channels, int px, int py, int channel);

  static void CopyPixelsToBuffer(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int src_width,
                                 int dst_width, int height, int channels, int src_start_col);

  static void CopyBufferToOutput(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int src_width,
                                 int dst_width, int height, int channels, int dst_start_col);
};

}  // namespace kondrashova_v_gauss_filter_vertical_split
