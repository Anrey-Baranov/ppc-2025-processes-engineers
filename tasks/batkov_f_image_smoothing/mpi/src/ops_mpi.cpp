#include "batkov_f_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "batkov_f_image_smoothing/common/include/common.hpp"

namespace batkov_f_image_smoothing {

namespace {

void CopyLocalImageData(const std::vector<uint8_t> &img_data, size_t width, size_t channels, size_t local_start,
                        size_t local_height, std::vector<uint8_t> &local_data) {
  for (size_t y_pos = 0; y_pos < local_height; y_pos++) {
    size_t global_y = local_start + y_pos;
    for (size_t x_pos = 0; x_pos < width; x_pos++) {
      for (size_t ch = 0; ch < channels; ch++) {
        local_data[((y_pos * width + x_pos) * channels) + ch] = img_data[((global_y * width + x_pos) * channels) + ch];
      }
    }
  }
}

void ProcessLocalImage(const std::vector<uint8_t> &local_data, const std::vector<std::vector<float>> &kernel,
                       size_t width, size_t channels, size_t start_row, size_t end_row, size_t local_start,
                       size_t local_height, size_t kernel_size, size_t half, std::vector<uint8_t> &local_result) {
  for (size_t y_local = 0; y_local < (end_row - start_row); y_local++) {
    size_t y_in_local_data = y_local + (start_row - local_start);
    for (size_t x_px = 0; x_px < width; x_px++) {
      for (size_t ch = 0; ch < channels; ch++) {
        float value = 0.0F;
        for (size_t ky = 0; ky < kernel_size; ky++) {
          for (size_t kx = 0; kx < kernel_size; kx++) {
            size_t px = x_px + kx - half;
            size_t py = y_in_local_data + ky - half;

            px = std::max<size_t>(0, std::min(px, width - 1));
            py = std::max<size_t>(0, std::min(py, local_height - 1));

            uint8_t pixel_value = local_data[((py * width + px) * channels) + ch];
            value += static_cast<float>(pixel_value) * kernel[ky][kx];
          }
        }
        local_result[((y_local * width + x_px) * channels) + ch] = static_cast<uint8_t>(value);
      }
    }
  }
}

void CopyProcess0Result(const std::vector<uint8_t> &local_result, size_t width, size_t channels, size_t start_row,
                        size_t end_row, std::vector<uint8_t> &result) {
  size_t process0_rows = end_row - start_row;
  for (size_t y_pos = 0; y_pos < process0_rows; y_pos++) {
    for (size_t x_pos = 0; x_pos < width; x_pos++) {
      for (size_t ch = 0; ch < channels; ch++) {
        result[(((start_row + y_pos) * width + x_pos) * channels) + ch] =
            local_result[((y_pos * width + x_pos) * channels) + ch];
      }
    }
  }
}

void GatherResultsFromProcesses(size_t size, size_t width, size_t channels, size_t rows_per_process, size_t remainder,
                                size_t start_row, size_t end_row, const std::vector<uint8_t> &local_result,
                                std::vector<uint8_t> &result) {
  CopyProcess0Result(local_result, width, channels, start_row, end_row, result);

  for (size_t proc = 1; proc < size; proc++) {
    size_t p_start = (proc * rows_per_process) + std::min<size_t>(proc, remainder);
    size_t p_end = p_start + rows_per_process + (proc < remainder ? 1 : 0);
    size_t p_rows = p_end - p_start;
    size_t p_data_size = width * p_rows * channels;

    std::vector<uint8_t> recv_buffer(p_data_size);
    MPI_Recv(recv_buffer.data(), static_cast<int>(p_data_size), MPI_UNSIGNED_CHAR, static_cast<int>(proc), 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (size_t y_pos = 0; y_pos < p_rows; y_pos++) {
      for (size_t x_pos = 0; x_pos < width; x_pos++) {
        for (size_t ch = 0; ch < channels; ch++) {
          result[(((p_start + y_pos) * width + x_pos) * channels) + ch] =
              recv_buffer[((y_pos * width + x_pos) * channels) + ch];
        }
      }
    }
  }
}

}  // namespace

BatkovFImageSmoothingMPI::BatkovFImageSmoothingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Image();
}

bool BatkovFImageSmoothingMPI::ValidationImpl() {
  return (!GetInput().data.empty()) && (GetInput().width > 0) && (GetInput().height > 0);
}

bool BatkovFImageSmoothingMPI::PreProcessingImpl() {
  size_t size = 5;
  float sigma = 1.0F;

  gaussian_kernel_.resize(size);
  for (auto &v : gaussian_kernel_) {
    v.resize(size);
  }

  float sum = 0.0F;
  size_t half = size / 2;

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      size_t x = i - half;
      size_t y = j - half;
      float value = std::exp((-static_cast<float>((x * x) + (y * y)) / (2 * sigma * sigma)));
      gaussian_kernel_[i][j] = value;
      sum += value;
    }
  }

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      gaussian_kernel_[i][j] /= sum;
    }
  }

  return true;
}

bool BatkovFImageSmoothingMPI::RunImpl() {
  int int_rank = 0;
  int int_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &int_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &int_size);

  auto &img = GetInput();
  size_t width = img.width;
  size_t height = img.height;
  size_t channels = img.channels;
  const auto &img_data = img.data;

  const auto rank = static_cast<size_t>(int_rank);
  const auto size = static_cast<size_t>(int_size);
  const size_t kernel_size = 5;
  const size_t half = kernel_size / 2;

  size_t rows_per_process = height / size;
  size_t remainder = height % size;
  size_t start_row = (rank * rows_per_process) + std::min<size_t>(rank, remainder);
  size_t end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

  size_t local_start = (start_row > half) ? start_row - half : 0;
  size_t local_end = (end_row + half < height) ? end_row + half : height;
  size_t local_height = local_end - local_start;

  std::vector<uint8_t> local_data(width * local_height * channels);
  CopyLocalImageData(img_data, width, channels, local_start, local_height, local_data);

  std::vector<uint8_t> local_result(width * (end_row - start_row) * channels);
  ProcessLocalImage(local_data, gaussian_kernel_, width, channels, start_row, end_row, local_start, local_height,
                    kernel_size, half, local_result);

  std::vector<uint8_t> result(width * height * channels);
  if (rank == 0) {
    GatherResultsFromProcesses(size, width, channels, rows_per_process, remainder, start_row, end_row, local_result,
                               result);
  } else {
    size_t data_size = width * (end_row - start_row) * channels;
    MPI_Send(local_result.data(), static_cast<int>(data_size), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Bcast(result.data(), static_cast<int>(width * height * channels), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  GetOutput().data = std::move(result);
  GetOutput().width = width;
  GetOutput().height = height;
  GetOutput().channels = channels;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool BatkovFImageSmoothingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace batkov_f_image_smoothing
