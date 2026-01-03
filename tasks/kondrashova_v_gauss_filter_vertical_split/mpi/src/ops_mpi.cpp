#include "kondrashova_v_gauss_filter_vertical_split/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

const std::array<std::array<int, 3>, 3> KondrashovaVGaussFilterVerticalSplitMPI::kGaussKernel = {
    {{{1, 2, 1}}, {{2, 4, 2}}, {{1, 2, 1}}}};
const int KondrashovaVGaussFilterVerticalSplitMPI::kGaussKernelSum = 16;

KondrashovaVGaussFilterVerticalSplitMPI::KondrashovaVGaussFilterVerticalSplitMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

uint8_t KondrashovaVGaussFilterVerticalSplitMPI::ApplyGaussToLocalPixel(const std::vector<uint8_t> &local_data,
                                                                        int local_width, int height, int channels,
                                                                        int px, int py, int channel) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = std::clamp(px + kx, 0, local_width - 1);
      int ny = std::clamp(py + ky, 0, height - 1);

      int idx = (((ny * local_width) + nx) * channels) + channel;
      auto kernel_row = static_cast<size_t>(ky) + 1;
      auto kernel_col = static_cast<size_t>(kx) + 1;
      sum += local_data[idx] * kGaussKernel.at(kernel_row).at(kernel_col);
    }
  }

  return static_cast<uint8_t>(std::clamp(sum / kGaussKernelSum, 0, 255));
}

bool KondrashovaVGaussFilterVerticalSplitMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();

    auto expected_size = static_cast<size_t>(input.width) * input.height * input.channels;
    return input.pixels.size() == expected_size && input.width >= 3 && input.height >= 3 && input.channels >= 1 &&
           input.channels <= 4;
  }
  return true;
}

bool KondrashovaVGaussFilterVerticalSplitMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();
    auto &output = GetOutput();

    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.pixels.resize(input.pixels.size());
  }
  return true;
}

void KondrashovaVGaussFilterVerticalSplitMPI::BroadcastImageDimensions(int &width, int &height, int &channels) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    width = GetInput().width;
    height = GetInput().height;
    channels = GetInput().channels;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void KondrashovaVGaussFilterVerticalSplitMPI::CalculateColumnDistribution(int width, int size,
                                                                          std::vector<int> &col_counts,
                                                                          std::vector<int> &col_offsets) {
  int base_cols = width / size;
  int extra_cols = width % size;

  col_counts.resize(size);
  col_offsets.resize(size);

  for (int i = 0; i < size; ++i) {
    col_counts[i] = base_cols + (i < extra_cols ? 1 : 0);
    col_offsets[i] = (i == 0) ? 0 : col_offsets[i - 1] + col_counts[i - 1];
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::CopyPixelsToBuffer(const std::vector<uint8_t> &src,
                                                                 std::vector<uint8_t> &dst, int src_width,
                                                                 int dst_width, int height, int channels,
                                                                 int src_start_col) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < dst_width; ++col) {
      for (int ch = 0; ch < channels; ++ch) {
        int src_idx = (((row * src_width) + (src_start_col + col)) * channels) + ch;
        int dst_idx = (((row * dst_width) + col) * channels) + ch;
        dst[dst_idx] = src[src_idx];
      }
    }
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::CopyBufferToOutput(const std::vector<uint8_t> &src,
                                                                 std::vector<uint8_t> &dst, int src_width,
                                                                 int dst_width, int height, int channels,
                                                                 int dst_start_col) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < src_width; ++col) {
      for (int ch = 0; ch < channels; ++ch) {
        int src_idx = (((row * src_width) + col) * channels) + ch;
        int dst_idx = (((row * dst_width) + (dst_start_col + col)) * channels) + ch;
        dst[dst_idx] = src[src_idx];
      }
    }
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::DistributeImageData(int rank, int size, int width, int height,
                                                                  int channels, const std::vector<int> &col_counts,
                                                                  const std::vector<int> &col_offsets,
                                                                  std::vector<uint8_t> &local_data, int extended_cols) {
  local_data.resize(static_cast<size_t>(extended_cols) * height * channels);

  if (rank == 0) {
    const auto &input_pixels = GetInput().pixels;

    for (int proc = 0; proc < size; ++proc) {
      int p_start = std::max(0, col_offsets[proc] - 1);
      int p_end = std::min(width, col_offsets[proc] + col_counts[proc] + 1);
      int p_cols = p_end - p_start;

      std::vector<uint8_t> send_data(static_cast<size_t>(p_cols) * height * channels);
      CopyPixelsToBuffer(input_pixels, send_data, width, p_cols, height, channels, p_start);

      if (proc == 0) {
        local_data = send_data;
      } else {
        MPI_Send(send_data.data(), static_cast<int>(send_data.size()), MPI_BYTE, proc, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Status status;
    MPI_Recv(local_data.data(), static_cast<int>(local_data.size()), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::ApplyGaussFilterToLocalData(const std::vector<uint8_t> &local_data,
                                                                          std::vector<uint8_t> &local_result,
                                                                          int extended_cols, int local_cols, int height,
                                                                          int channels, int offset_in_extended) {
  local_result.resize(static_cast<size_t>(local_cols) * height * channels);

  for (int row = 0; row < height; ++row) {
    for (int lx = 0; lx < local_cols; ++lx) {
      int col = offset_in_extended + lx;
      for (int ch = 0; ch < channels; ++ch) {
        int result_idx = (((row * local_cols) + lx) * channels) + ch;
        local_result[result_idx] = ApplyGaussToLocalPixel(local_data, extended_cols, height, channels, col, row, ch);
      }
    }
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::GatherResults(int rank, int size, int width, int height, int channels,
                                                            const std::vector<int> &col_counts,
                                                            const std::vector<int> &col_offsets, int local_start_col,
                                                            int local_cols, const std::vector<uint8_t> &local_result) {
  if (rank == 0) {
    auto &output_pixels = GetOutput().pixels;

    CopyBufferToOutput(local_result, output_pixels, local_cols, width, height, channels, local_start_col);

    for (int proc = 1; proc < size; ++proc) {
      int p_cols = col_counts[proc];
      std::vector<uint8_t> recv_data(static_cast<size_t>(p_cols) * height * channels);
      MPI_Status status;
      MPI_Recv(recv_data.data(), static_cast<int>(recv_data.size()), MPI_BYTE, proc, 1, MPI_COMM_WORLD, &status);

      CopyBufferToOutput(recv_data, output_pixels, p_cols, width, height, channels, col_offsets[proc]);
    }
  } else {
    MPI_Send(local_result.data(), static_cast<int>(local_result.size()), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
  }
}

void KondrashovaVGaussFilterVerticalSplitMPI::BroadcastResultToAllProcesses(int width, int height, int channels) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    auto &output = GetOutput();
    output.width = width;
    output.height = height;
    output.channels = channels;
    output.pixels.resize(static_cast<size_t>(width) * height * channels);
  }

  MPI_Bcast(GetOutput().pixels.data(), static_cast<int>(GetOutput().pixels.size()), MPI_BYTE, 0, MPI_COMM_WORLD);
}

bool KondrashovaVGaussFilterVerticalSplitMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int width = 0;
  int height = 0;
  int channels = 0;
  BroadcastImageDimensions(width, height, channels);

  std::vector<int> col_counts;
  std::vector<int> col_offsets;
  CalculateColumnDistribution(width, size, col_counts, col_offsets);

  int local_start_col = col_offsets[rank];
  int local_cols = col_counts[rank];

  int extended_start = std::max(0, local_start_col - 1);
  int extended_end = std::min(width, local_start_col + local_cols + 1);
  int extended_cols = extended_end - extended_start;
  int offset_in_extended = local_start_col - extended_start;

  std::vector<uint8_t> local_data;
  DistributeImageData(rank, size, width, height, channels, col_counts, col_offsets, local_data, extended_cols);

  std::vector<uint8_t> local_result;
  ApplyGaussFilterToLocalData(local_data, local_result, extended_cols, local_cols, height, channels,
                              offset_in_extended);

  GatherResults(rank, size, width, height, channels, col_counts, col_offsets, local_start_col, local_cols,
                local_result);

  BroadcastResultToAllProcesses(width, height, channels);

  return true;
}

bool KondrashovaVGaussFilterVerticalSplitMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kondrashova_v_gauss_filter_vertical_split
