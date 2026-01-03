#include "shilin_n_gauss_filter_vertical_split/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "shilin_n_gauss_filter_vertical_split/common/include/common.hpp"

namespace shilin_n_gauss_filter_vertical_split {

ShilinNGaussFilterVerticalSplitMPI::ShilinNGaussFilterVerticalSplitMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = std::vector<uint8_t>();
}

bool ShilinNGaussFilterVerticalSplitMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int validation_result = 1;

  if (rank == 0) {
    const InType &input = GetInput();
    if (input.width <= 0 || input.height <= 0 || input.channels <= 0) {
      validation_result = 0;
    } else {
      size_t expected_size =
          static_cast<size_t>(input.width) * static_cast<size_t>(input.height) * static_cast<size_t>(input.channels);
      if (input.pixels.size() != expected_size) {
        validation_result = 0;
      }
    }
  }

  MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return validation_result != 0;
}

bool ShilinNGaussFilterVerticalSplitMPI::PreProcessingImpl() {
  GetOutput() = std::vector<uint8_t>();
  return true;
}

bool ShilinNGaussFilterVerticalSplitMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int width = 0;
  int height = 0;
  int channels = 0;

  std::vector<uint8_t> input_pixels;

  if (rank == 0) {
    const InType &input = GetInput();
    width = input.width;
    height = input.height;
    channels = input.channels;
    input_pixels = input.pixels;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (width <= 0 || height <= 0 || channels <= 0) {
    return false;
  }

  int local_width = 0;
  int local_start_col = 0;

  std::vector<uint8_t> local_input;
  DistributeVerticalStripes(input_pixels, local_input, width, height, channels, rank, size, local_width,
                            local_start_col);

  std::vector<uint8_t> local_output(static_cast<size_t>(local_width) * static_cast<size_t>(height) *
                                    static_cast<size_t>(channels));
  ApplyGaussianKernelMPI(local_input, local_output, local_width, local_start_col, width, height, channels);

  std::vector<uint8_t> output_pixels;
  if (rank == 0) {
    output_pixels =
        std::vector<uint8_t>(static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels));
  }

  GatherVerticalStripes(local_output, output_pixels, width, height, channels, rank, size, local_width, local_start_col);

  const auto output_size = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
  if (rank != 0) {
    output_pixels.resize(output_size);
  }

  // синхронизируем итоговое изображение между всеми процессами, чтобы тесты не падали на worker ranks
  MPI_Bcast(output_pixels.data(), static_cast<int>(output_size), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  GetOutput() = output_pixels;

  return true;
}

bool ShilinNGaussFilterVerticalSplitMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return (rank == 0) ? !GetOutput().empty() : true;
}

void ShilinNGaussFilterVerticalSplitMPI::DistributeVerticalStripes(const std::vector<uint8_t> &source_image,
                                                                   std::vector<uint8_t> &destination_stripe, int width,
                                                                   int height, int channels, int rank, int size,
                                                                   int &local_width, int &local_start_col) {
  // вертикальное разбиение: каждый процесс получает несколько столбцов
  int base_cols_per_proc = width / size;
  int remainder = width % size;

  local_start_col = (rank * base_cols_per_proc) + std::min(rank, remainder);
  local_width = base_cols_per_proc + (rank < remainder ? 1 : 0);

  // для фильтра 3x3 нужны граничные пиксели слева и справа
  int left_padding = (local_start_col > 0) ? 1 : 0;
  int right_padding = (local_start_col + local_width < width) ? 1 : 0;
  int extended_width = local_width + left_padding + right_padding;

  size_t local_size = static_cast<size_t>(extended_width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
  destination_stripe = std::vector<uint8_t>(local_size);

  if (rank == 0) {
    // процесс 0 отправляет данные остальным процессам
    for (int dest = 1; dest < size; ++dest) {
      SendDataToProcess(source_image, dest, width, height, channels, base_cols_per_proc, remainder);
    }

    // процесс 0 обрабатывает свои данные
    CopyLocalData(source_image, destination_stripe, local_start_col, local_width, width, height, channels);
  } else {
    // остальные процессы получают данные
    MPI_Recv(destination_stripe.data(), static_cast<int>(destination_stripe.size()), MPI_UNSIGNED_CHAR, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void ShilinNGaussFilterVerticalSplitMPI::SendDataToProcess(const std::vector<uint8_t> &input, int dest, int width,
                                                           int height, int channels, int base_cols_per_proc,
                                                           int remainder) {
  int dest_start = (dest * base_cols_per_proc) + std::min(dest, remainder);
  int dest_width = base_cols_per_proc + (dest < remainder ? 1 : 0);
  int dest_left_padding = (dest_start > 0) ? 1 : 0;
  int dest_right_padding = (dest_start + dest_width < width) ? 1 : 0;
  int dest_extended_width = dest_width + dest_left_padding + dest_right_padding;

  int send_start_col = dest_start - dest_left_padding;
  int send_width = dest_extended_width;

  std::vector<uint8_t> send_data(static_cast<size_t>(send_width) * static_cast<size_t>(height) *
                                 static_cast<size_t>(channels));
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < send_width; ++col) {
      int src_x = send_start_col + col;
      if (src_x >= 0 && src_x < width) {
        for (int ch = 0; ch < channels; ++ch) {
          size_t src_idx = (static_cast<size_t>(row) * static_cast<size_t>(width) * static_cast<size_t>(channels)) +
                           (static_cast<size_t>(src_x) * static_cast<size_t>(channels)) + static_cast<size_t>(ch);
          size_t dst_idx =
              (static_cast<size_t>(row) * static_cast<size_t>(send_width) * static_cast<size_t>(channels)) +
              (static_cast<size_t>(col) * static_cast<size_t>(channels)) + static_cast<size_t>(ch);
          send_data[dst_idx] = input[src_idx];
        }
      }
    }
  }

  MPI_Send(send_data.data(), static_cast<int>(send_data.size()), MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
}

void ShilinNGaussFilterVerticalSplitMPI::CopyLocalData(const std::vector<uint8_t> &input,
                                                       std::vector<uint8_t> &local_data, int local_start_col,
                                                       int local_width, int width, int height, int channels) {
  int left_padding = (local_start_col > 0) ? 1 : 0;
  int right_padding = (local_start_col + local_width < width) ? 1 : 0;
  int extended_width = local_width + left_padding + right_padding;

  for (int row = 0; row < height; ++row) {
    for (int col = -left_padding; col < local_width + right_padding; ++col) {
      int src_x = local_start_col + col;
      if (src_x >= 0 && src_x < width) {
        for (int ch = 0; ch < channels; ++ch) {
          size_t src_idx = (static_cast<size_t>(row) * static_cast<size_t>(width) * static_cast<size_t>(channels)) +
                           (static_cast<size_t>(src_x) * static_cast<size_t>(channels)) + static_cast<size_t>(ch);
          size_t dst_idx =
              (static_cast<size_t>(row) * static_cast<size_t>(extended_width) * static_cast<size_t>(channels)) +
              (static_cast<size_t>(col + left_padding) * static_cast<size_t>(channels)) + static_cast<size_t>(ch);
          local_data[dst_idx] = input[src_idx];
        }
      }
    }
  }
}

void ShilinNGaussFilterVerticalSplitMPI::ApplyGaussianKernelMPI(const std::vector<uint8_t> &local_input,
                                                                std::vector<uint8_t> &local_output, int local_width,
                                                                int local_start_col, int width, int height,
                                                                int channels) {
  int left_padding = (local_start_col > 0) ? 1 : 0;
  int right_padding = (local_start_col + local_width < width) ? 1 : 0;
  int extended_width = local_width + left_padding + right_padding;

  for (int row = 0; row < height; ++row) {
    for (int local_col = 0; local_col < local_width; ++local_col) {
      ProcessPixelWithKernel(local_input, local_output, row, local_col, local_width, left_padding, extended_width,
                             height, channels);
    }
  }
}

void ShilinNGaussFilterVerticalSplitMPI::ProcessPixelWithKernel(const std::vector<uint8_t> &local_input,
                                                                std::vector<uint8_t> &local_output, int row,
                                                                int local_col, int local_width, int left_padding,
                                                                int extended_width, int height, int channels) {
  // ядро гаусса 3x3
  constexpr std::array<std::array<double, 3>, 3> kKernel = {{{{1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0}},
                                                             {{2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0}},
                                                             {{1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0}}}};

  int col_in_extended = local_col + left_padding;

  for (int ch = 0; ch < channels; ++ch) {
    double sum = 0.0;

    for (int ky = -1; ky <= 1; ++ky) {
      for (int kx = -1; kx <= 1; ++kx) {
        int px = col_in_extended + kx;
        int py = row + ky;

        double pixel_val = 0.0;
        if (px >= 0 && px < extended_width && py >= 0 && py < height) {
          size_t idx = (static_cast<size_t>(py) * static_cast<size_t>(extended_width) * static_cast<size_t>(channels)) +
                       (static_cast<size_t>(px) * static_cast<size_t>(channels)) + static_cast<size_t>(ch);
          pixel_val = static_cast<double>(local_input[idx]);
        }
        const int kernel_y_idx = ky + 1;
        const int kernel_x_idx = kx + 1;
        const auto kernel_y = static_cast<size_t>(kernel_y_idx);
        const auto kernel_x = static_cast<size_t>(kernel_x_idx);
        sum += pixel_val * kKernel.at(kernel_y).at(kernel_x);
      }
    }

    size_t out_idx = (static_cast<size_t>(row) * static_cast<size_t>(local_width) * static_cast<size_t>(channels)) +
                     (static_cast<size_t>(local_col) * static_cast<size_t>(channels)) + static_cast<size_t>(ch);
    local_output[out_idx] = static_cast<uint8_t>(std::clamp(sum, 0.0, 255.0));
  }
}

void ShilinNGaussFilterVerticalSplitMPI::GatherVerticalStripes(const std::vector<uint8_t> &local_stripe,
                                                               std::vector<uint8_t> &final_image, int width, int height,
                                                               int channels, int rank, int size, int local_width,
                                                               int local_start_col) {
  int base_cols_per_proc = width / size;
  int remainder = width % size;

  if (rank == 0) {
    int src_start = (0 * base_cols_per_proc) + std::min(0, remainder);
    int src_width = base_cols_per_proc + (0 < remainder ? 1 : 0);
    GatherFromRank0(local_stripe, final_image, width, height, channels, size, local_width, src_start, src_width);
    GatherFromOtherRanks(final_image, width, height, channels, size, base_cols_per_proc, remainder);
  } else {
    SendUnpaddedData(local_stripe, local_width, local_start_col, width, height, channels);
  }
}

void ShilinNGaussFilterVerticalSplitMPI::GatherFromRank0(const std::vector<uint8_t> &local_stripe,
                                                         std::vector<uint8_t> &final_image, int width, int height,
                                                         int channels, int /* size */, int local_width, int src_start,
                                                         int src_width) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < src_width; ++col) {
      for (int ch = 0; ch < channels; ++ch) {
        size_t local_idx =
            (static_cast<size_t>(row) * static_cast<size_t>(local_width) * static_cast<size_t>(channels)) +
            (static_cast<size_t>(col) * static_cast<size_t>(channels)) + static_cast<size_t>(ch);
        size_t global_idx = (static_cast<size_t>(row) * static_cast<size_t>(width) * static_cast<size_t>(channels)) +
                            (static_cast<size_t>(src_start + col) * static_cast<size_t>(channels)) +
                            static_cast<size_t>(ch);
        final_image[global_idx] = local_stripe[local_idx];
      }
    }
  }
}

void ShilinNGaussFilterVerticalSplitMPI::GatherFromOtherRanks(std::vector<uint8_t> &final_image, int width, int height,
                                                              int channels, int size, int base_cols_per_proc,
                                                              int remainder) {
  for (int src = 1; src < size; ++src) {
    int src_start = (src * base_cols_per_proc) + std::min(src, remainder);
    int src_width = base_cols_per_proc + (src < remainder ? 1 : 0);

    std::vector<uint8_t> recv_data(static_cast<size_t>(src_width) * static_cast<size_t>(height) *
                                   static_cast<size_t>(channels));

    MPI_Recv(recv_data.data(), static_cast<int>(recv_data.size()), MPI_UNSIGNED_CHAR, src, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < src_width; ++col) {
        for (int ch = 0; ch < channels; ++ch) {
          size_t recv_idx =
              (static_cast<size_t>(row) * static_cast<size_t>(src_width) * static_cast<size_t>(channels)) +
              (static_cast<size_t>(col) * static_cast<size_t>(channels)) + static_cast<size_t>(ch);
          size_t global_idx = (static_cast<size_t>(row) * static_cast<size_t>(width) * static_cast<size_t>(channels)) +
                              (static_cast<size_t>(src_start + col) * static_cast<size_t>(channels)) +
                              static_cast<size_t>(ch);
          final_image[global_idx] = recv_data[recv_idx];
        }
      }
    }
  }
}

void ShilinNGaussFilterVerticalSplitMPI::SendUnpaddedData(const std::vector<uint8_t> &local_stripe, int local_width,
                                                          int /*local_start_col*/, int /*width*/, int height,
                                                          int channels) {
  // local_stripe уже содержит unpadded данные (local_output после ApplyGaussianKernelMPI)
  // размер: local_width * height * channels
  size_t data_size = static_cast<size_t>(local_width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
  MPI_Send(local_stripe.data(), static_cast<int>(data_size), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
}

}  // namespace shilin_n_gauss_filter_vertical_split
