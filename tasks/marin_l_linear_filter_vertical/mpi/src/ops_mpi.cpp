#include "marin_l_linear_filter_vertical/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "marin_l_linear_filter_vertical/common/include/common.hpp"

namespace marin_l_linear_filter_vertical {

namespace {
constexpr std::array<std::array<int, 3>, 3> kGaussKernel = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
constexpr int kKernelSum = 16;

uint8_t ApplyKernelLocal(const std::vector<uint8_t> &local_input, int ext_col_count, int height, int row, int lx,
                         int left_pad) {
  int sum = 0;
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int ny = row + ky;
      int local_nx = lx + left_pad + kx;
      uint8_t pixel_value = 0;
      if (ny >= 0 && ny < height && local_nx >= 0 && local_nx < ext_col_count) {
        pixel_value = local_input[(ny * ext_col_count) + local_nx];
      }
      sum += pixel_value * kGaussKernel.at(ky + 1).at(kx + 1);
    }
  }
  return static_cast<uint8_t>(std::clamp(sum / kKernelSum, 0, 255));
}

void ComputeColDistribution(int width, int size, std::vector<int> &col_counts, std::vector<int> &col_starts) {
  int cols_per_proc = width / size;
  int extra_cols = width % size;
  int current_col = 0;
  for (int proc = 0; proc < size; ++proc) {
    col_starts[proc] = current_col;
    col_counts[proc] = cols_per_proc + ((proc < extra_cols) ? 1 : 0);
    current_col += col_counts[proc];
  }
}

void ExtractLocalData(const std::vector<uint8_t> &input_pixels, std::vector<uint8_t> &local_input, int width,
                      int height, int ext_start, int ext_col_count) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < ext_col_count; ++col) {
      local_input[(row * ext_col_count) + col] = input_pixels[(row * width) + (ext_start + col)];
    }
  }
}

void CopyLocalToOutput(std::vector<uint8_t> &output_pixels, const std::vector<uint8_t> &local_output, int width,
                       int height, int col_start, int col_count) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < col_count; ++col) {
      output_pixels[(row * width) + (col_start + col)] = local_output[(row * col_count) + col];
    }
  }
}

int ComputePadding(int col_start, int col_count, int width, bool is_left) {
  if (is_left) {
    return (col_start > 0) ? 1 : 0;
  }
  return ((col_start + col_count) < width) ? 1 : 0;
}

void SendDataToWorkers(const std::vector<uint8_t> &input_pixels, const std::vector<int> &col_counts,
                       const std::vector<int> &col_starts, int width, int height, int size) {
  for (int proc = 1; proc < size; ++proc) {
    if (col_counts[proc] == 0) {
      continue;
    }
    int p_col_start = col_starts[proc];
    int p_col_count = col_counts[proc];
    int p_left_pad = ComputePadding(p_col_start, p_col_count, width, true);
    int p_right_pad = ComputePadding(p_col_start, p_col_count, width, false);
    int p_ext_col_count = p_col_count + p_left_pad + p_right_pad;
    int p_ext_start = p_col_start - p_left_pad;

    std::vector<uint8_t> send_buf(static_cast<size_t>(p_ext_col_count) * static_cast<size_t>(height));
    ExtractLocalData(input_pixels, send_buf, width, height, p_ext_start, p_ext_col_count);
    MPI_Send(send_buf.data(), static_cast<int>(send_buf.size()), MPI_UNSIGNED_CHAR, proc, 0, MPI_COMM_WORLD);
  }
}

void ReceiveResultsFromWorkers(std::vector<uint8_t> &output_pixels, const std::vector<int> &col_counts,
                               const std::vector<int> &col_starts, int width, int height, int size) {
  for (int proc = 1; proc < size; ++proc) {
    if (col_counts[proc] == 0) {
      continue;
    }
    int p_col_start = col_starts[proc];
    int p_col_count = col_counts[proc];
    std::vector<uint8_t> recv_buf(static_cast<size_t>(p_col_count) * static_cast<size_t>(height));
    MPI_Recv(recv_buf.data(), static_cast<int>(recv_buf.size()), MPI_UNSIGNED_CHAR, proc, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    CopyLocalToOutput(output_pixels, recv_buf, width, height, p_col_start, p_col_count);
  }
}

void ApplyFilterToLocalData(const std::vector<uint8_t> &local_input, std::vector<uint8_t> &local_output,
                            int ext_col_count, int local_col_count, int height, int left_pad) {
  for (int row = 0; row < height; ++row) {
    for (int lx = 0; lx < local_col_count; ++lx) {
      local_output[(row * local_col_count) + lx] =
          ApplyKernelLocal(local_input, ext_col_count, height, row, lx, left_pad);
    }
  }
}
}  // namespace

MarinLLinearFilterVerticalMPI::MarinLLinearFilterVerticalMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool MarinLLinearFilterVerticalMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &input = GetInput();
    if (input.width <= 0 || input.height <= 0) {
      return false;
    }
    auto expected_size = static_cast<size_t>(input.width) * static_cast<size_t>(input.height);
    return input.pixels.size() == expected_size;
  }
  return true;
}

bool MarinLLinearFilterVerticalMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &input = GetInput();
    width_ = input.width;
    height_ = input.height;
    input_pixels_ = input.pixels;
  }
  return true;
}

bool MarinLLinearFilterVerticalMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  output_pixels_.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_));

  std::vector<int> col_counts(size);
  std::vector<int> col_starts(size);
  ComputeColDistribution(width_, size, col_counts, col_starts);

  int local_col_start = col_starts[rank];
  int local_col_count = col_counts[rank];

  if (local_col_count == 0) {
    MPI_Bcast(output_pixels_.data(), static_cast<int>(output_pixels_.size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    return true;
  }

  int left_pad = ComputePadding(local_col_start, local_col_count, width_, true);
  int right_pad = ComputePadding(local_col_start, local_col_count, width_, false);
  int extended_col_count = local_col_count + left_pad + right_pad;

  std::vector<uint8_t> local_input(static_cast<size_t>(extended_col_count) * static_cast<size_t>(height_));

  if (rank == 0) {
    int ext_start = local_col_start - left_pad;
    ExtractLocalData(input_pixels_, local_input, width_, height_, ext_start, extended_col_count);
    SendDataToWorkers(input_pixels_, col_counts, col_starts, width_, height_, size);
  } else {
    MPI_Recv(local_input.data(), static_cast<int>(local_input.size()), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  std::vector<uint8_t> local_output(static_cast<size_t>(local_col_count) * static_cast<size_t>(height_));
  ApplyFilterToLocalData(local_input, local_output, extended_col_count, local_col_count, height_, left_pad);

  if (rank == 0) {
    CopyLocalToOutput(output_pixels_, local_output, width_, height_, local_col_start, local_col_count);
    ReceiveResultsFromWorkers(output_pixels_, col_counts, col_starts, width_, height_, size);
  } else {
    MPI_Send(local_output.data(), static_cast<int>(local_output.size()), MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD);
  }

  MPI_Bcast(output_pixels_.data(), static_cast<int>(output_pixels_.size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  return true;
}

bool MarinLLinearFilterVerticalMPI::PostProcessingImpl() {
  GetOutput() = output_pixels_;
  return true;
}

}  // namespace marin_l_linear_filter_vertical
