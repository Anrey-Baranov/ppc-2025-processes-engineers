#include "borunov_v_block_partitioning/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "borunov_v_block_partitioning/common/include/common.hpp"

namespace borunov_v_block_partitioning {

BorunovVBlockPartitioningMPI::BorunovVBlockPartitioningMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BorunovVBlockPartitioningMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (GetInput().size() < 3) {
      return false;
    }
    int w = GetInput()[0];
    int h = GetInput()[1];
    const std::size_t pixels = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    const std::size_t expected_size = static_cast<std::size_t>(2) + pixels;
    return GetInput().size() == expected_size;
  }
  return true;
}

bool BorunovVBlockPartitioningMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    int w = GetInput()[0];
    int h = GetInput()[1];
    GetOutput().assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  }
  return true;
}

namespace {

void BroadcastDims(int &width, int &height) {
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void ComputeSendCountsDispls(int width, int height, int size, std::vector<int> &send_counts, std::vector<int> &displs) {
  send_counts.assign(static_cast<std::size_t>(size), 0);
  displs.assign(static_cast<std::size_t>(size), 0);

  const int rows_per_proc = height / size;
  const int remainder = height % size;
  int current_displ = 0;
  for (int i = 0; i < size; ++i) {
    const int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
    send_counts[i] = proc_rows * width;
    displs[i] = current_displ;
    current_displ += send_counts[i];
  }
}

void ComputeSendCountsDisplsWithHalo(int width, int height, const std::vector<int> &send_counts,
                                     const std::vector<int> &displs, std::vector<int> &send_counts_halo,
                                     std::vector<int> &displs_halo) {
  const int size = static_cast<int>(send_counts.size());
  send_counts_halo.assign(static_cast<std::size_t>(size), 0);
  displs_halo.assign(static_cast<std::size_t>(size), 0);

  for (int i = 0; i < size; ++i) {
    const int row_start = displs[i] / width;
    const int rows = send_counts[i] / width;
    const int start_with_halo = std::max(0, row_start - 1);
    const int end_with_halo = std::min(height, row_start + rows + 1);
    send_counts_halo[i] = (end_with_halo - start_with_halo) * width;
    displs_halo[i] = start_with_halo * width;
  }
}

void ApplyKernelToLocalPartition(const int *local_pixels, int width, int height, int base_global_row,
                                 int global_row_start, int global_row_end, std::vector<int> &local_res) {
  const std::array<std::array<float, 3>, 3> kernel = {{
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
      {2.0F / 16.0F, 4.0F / 16.0F, 2.0F / 16.0F},
      {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F},
  }};

  for (int gi = global_row_start; gi < global_row_end; ++gi) {
    const int local_i = gi - global_row_start;
    for (int j = 0; j < width; ++j) {
      const int x0 = std::clamp(j - 1, 0, width - 1);
      const int x1 = j;
      const int x2 = std::clamp(j + 1, 0, width - 1);

      const int y0 = std::clamp(gi - 1, 0, height - 1);
      const int y1 = gi;
      const int y2 = std::clamp(gi + 1, 0, height - 1);

      const int y0_local = y0 - base_global_row;
      const int y1_local = y1 - base_global_row;
      const int y2_local = y2 - base_global_row;

      float sum = 0.0F;

      sum += static_cast<float>(local_pixels[(y0_local * width) + x0]) * kernel[0][0];
      sum += static_cast<float>(local_pixels[(y0_local * width) + x1]) * kernel[0][1];
      sum += static_cast<float>(local_pixels[(y0_local * width) + x2]) * kernel[0][2];

      sum += static_cast<float>(local_pixels[(y1_local * width) + x0]) * kernel[1][0];
      sum += static_cast<float>(local_pixels[(y1_local * width) + x1]) * kernel[1][1];
      sum += static_cast<float>(local_pixels[(y1_local * width) + x2]) * kernel[1][2];

      sum += static_cast<float>(local_pixels[(y2_local * width) + x0]) * kernel[2][0];
      sum += static_cast<float>(local_pixels[(y2_local * width) + x1]) * kernel[2][1];
      sum += static_cast<float>(local_pixels[(y2_local * width) + x2]) * kernel[2][2];

      local_res[(local_i * width) + j] = static_cast<int>(std::round(sum));
    }
  }
}

}  // namespace

bool BorunovVBlockPartitioningMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = 0;
  int height = 0;
  if (rank == 0) {
    width = GetInput()[0];
    height = GetInput()[1];
  }
  BroadcastDims(width, height);

  std::vector<int> send_counts;
  std::vector<int> displs;
  ComputeSendCountsDispls(width, height, size, send_counts, displs);

  std::vector<int> send_counts_halo;
  std::vector<int> displs_halo;
  ComputeSendCountsDisplsWithHalo(width, height, send_counts, displs, send_counts_halo, displs_halo);

  const int local_pixels_count = send_counts_halo[rank];
  std::vector<int> local_pixels(static_cast<std::size_t>(local_pixels_count));

  MPI_Scatterv(rank == 0 ? (GetInput().data() + 2) : nullptr, send_counts_halo.data(), displs_halo.data(), MPI_INT,
               local_pixels.data(), local_pixels_count, MPI_INT, 0, MPI_COMM_WORLD);

  const int my_rows = send_counts[rank] / width;
  const int row_start = displs[rank] / width;
  const int row_end = row_start + my_rows;

  const int base_global_row = displs_halo[rank] / width;

  std::vector<int> local_res(static_cast<std::size_t>(my_rows) * static_cast<std::size_t>(width));
  ApplyKernelToLocalPartition(local_pixels.data(), width, height, base_global_row, row_start, row_end, local_res);

  const int local_count = static_cast<int>(local_res.size());
  MPI_Gatherv(local_res.data(), local_count, MPI_INT, rank == 0 ? GetOutput().data() : nullptr, send_counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool BorunovVBlockPartitioningMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = 0;
  int height = 0;

  if (rank == 0) {
    width = GetInput()[0];
    height = GetInput()[1];
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int total_pixels = width * height;
  if (rank != 0) {
    GetOutput().assign(static_cast<std::size_t>(total_pixels), 0);
  }

  MPI_Bcast(GetOutput().data(), total_pixels, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

}  // namespace borunov_v_block_partitioning
