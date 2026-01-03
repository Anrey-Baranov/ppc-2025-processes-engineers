#include "vlasova_a_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "vlasova_a_image_smoothing/common/include/common.hpp"

namespace vlasova_a_image_smoothing {

namespace {

std::uint8_t ComputePixelMedian(int col_idx, int row_idx, int overlap_start,
                                const std::vector<std::uint8_t> &local_data, int width, int height, int window_size) {
  const int radius = window_size / 2;
  std::vector<std::uint8_t> neighbors;
  neighbors.reserve(static_cast<std::size_t>(window_size) * window_size);

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const int neighbor_x = col_idx + dx;
      const int neighbor_y = row_idx + dy;

      if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
        const int local_row = neighbor_y - overlap_start;
        const std::size_t index = (static_cast<std::size_t>(local_row) * width) + neighbor_x;
        neighbors.push_back(local_data[index]);
      }
    }
  }

  if (!neighbors.empty()) {
    auto middle = std::next(neighbors.begin(), static_cast<std::ptrdiff_t>(neighbors.size() / 2));
    std::nth_element(neighbors.begin(), middle, neighbors.end());
    return *middle;
  }

  const int local_row = row_idx - overlap_start;
  const std::size_t index = (static_cast<std::size_t>(local_row) * width) + col_idx;
  return local_data[index];
}

void PrepareScatterData(int size, int width, int height, int radius, std::vector<int> &sendcounts,
                        std::vector<int> &displs) {
  const int base_rows = height / size;
  const int extra_rows = height % size;

  for (int proc = 0; proc < size; ++proc) {
    const int proc_start = (proc * base_rows) + std::min(proc, extra_rows);
    const int proc_end = proc_start + base_rows + (proc < extra_rows ? 1 : 0);

    if (proc_end > proc_start) {
      const int proc_overlap_start = std::max(0, proc_start - radius);
      const int proc_overlap_end = std::min(height, proc_end + radius);
      sendcounts[static_cast<std::size_t>(proc)] = (proc_overlap_end - proc_overlap_start) * width;
      displs[static_cast<std::size_t>(proc)] = proc_overlap_start * width;
    } else {
      sendcounts[static_cast<std::size_t>(proc)] = 0;
      displs[static_cast<std::size_t>(proc)] = 0;
    }
  }
}

void PrepareGatherData(int size, int width, int height, std::vector<int> &sendcounts, std::vector<int> &displs) {
  const int base_rows = height / size;
  const int extra_rows = height % size;

  for (int proc = 0; proc < size; ++proc) {
    const int proc_start = (proc * base_rows) + std::min(proc, extra_rows);
    const int proc_end = proc_start + base_rows + (proc < extra_rows ? 1 : 0);
    sendcounts[static_cast<std::size_t>(proc)] = (proc_end - proc_start) * width;
    displs[static_cast<std::size_t>(proc)] = proc_start * width;
  }
}

}  // namespace

VlasovaAImageSmoothingMPI::VlasovaAImageSmoothingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VlasovaAImageSmoothingMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();

    if (input.width <= 0 || input.height <= 0) {
      return false;
    }

    if (input.data.empty()) {
      return false;
    }

    const std::size_t expected_size = static_cast<std::size_t>(input.width) * input.height;
    if (input.data.size() != expected_size) {
      return false;
    }
  }

  int is_valid = 1;
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return is_valid == 1;
}

bool VlasovaAImageSmoothingMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();
    width_ = input.width;
    height_ = input.height;
  }

  std::array<int, 2> dimensions = {width_, height_};
  MPI_Bcast(dimensions.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&window_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  width_ = dimensions[0];
  height_ = dimensions[1];

  return true;
}

bool VlasovaAImageSmoothingMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int radius = window_size_ / 2;
  const int base_rows = height_ / size;
  const int extra_rows = height_ % size;

  const int start_row = (rank * base_rows) + std::min(rank, extra_rows);
  const int end_row = start_row + base_rows + (rank < extra_rows ? 1 : 0);
  const int local_height = end_row - start_row;

  if (local_height <= 0) {
    GetOutput().width = width_;
    GetOutput().height = height_;
    GetOutput().data.clear();
    return true;
  }

  const int overlap_start = std::max(0, start_row - radius);
  const int overlap_end = std::min(height_, end_row + radius);
  const int overlap_height = overlap_end - overlap_start;

  std::vector<std::uint8_t> full_image;
  if (rank == 0) {
    full_image = GetInput().data;
  }

  std::vector<int> sendcounts(static_cast<std::size_t>(size));
  std::vector<int> displs(static_cast<std::size_t>(size));

  if (rank == 0) {
    PrepareScatterData(size, width_, height_, radius, sendcounts, displs);
  }

  std::vector<std::uint8_t> local_image(static_cast<std::size_t>(overlap_height) * width_);
  MPI_Scatterv((rank == 0 ? full_image.data() : nullptr), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
               local_image.data(), overlap_height * width_, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  std::vector<std::uint8_t> local_result(static_cast<std::size_t>(local_height) * width_);
  for (int local_row = 0; local_row < local_height; ++local_row) {
    const int global_row = start_row + local_row;

    for (int col_idx = 0; col_idx < width_; ++col_idx) {
      const std::size_t index = (static_cast<std::size_t>(local_row) * width_) + col_idx;
      local_result[index] =
          ComputePixelMedian(col_idx, global_row, overlap_start, local_image, width_, height_, window_size_);
    }
  }

  std::vector<std::uint8_t> result_image;
  if (rank == 0) {
    result_image.resize(static_cast<std::size_t>(width_) * height_);
  }

  if (rank == 0) {
    PrepareGatherData(size, width_, height_, sendcounts, displs);
  }

  MPI_Gatherv(local_result.data(), local_height * width_, MPI_UNSIGNED_CHAR,
              (rank == 0 ? result_image.data() : nullptr), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput().width = width_;
    GetOutput().height = height_;
    GetOutput().data = result_image;
  } else {
    GetOutput().width = width_;
    GetOutput().height = height_;
  }

  int data_size = static_cast<int>(GetOutput().data.size());
  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().data.resize(static_cast<std::size_t>(data_size));
  }

  MPI_Bcast(GetOutput().data.data(), data_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  return true;
}

bool VlasovaAImageSmoothingMPI::PostProcessingImpl() {
  const auto &output = GetOutput();

  if (output.data.empty()) {
    return false;
  }

  const std::size_t expected_size = static_cast<std::size_t>(width_) * height_;
  return output.data.size() == expected_size;
}

}  // namespace vlasova_a_image_smoothing
