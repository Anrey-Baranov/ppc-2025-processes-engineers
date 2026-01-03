#include "tsibareva_e_edge_select_sobel/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "tsibareva_e_edge_select_sobel/common/include/common.hpp"

namespace tsibareva_e_edge_select_sobel {

const std::vector<std::vector<int>> kSobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

const std::vector<std::vector<int>> kSobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

TsibarevaEEdgeSelectSobelMPI::TsibarevaEEdgeSelectSobelMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
    GetInput() = in;
    input_pixels_ = std::get<0>(GetInput());
    height_ = std::get<1>(GetInput());
    width_ = std::get<2>(GetInput());
    threshold_ = std::get<3>(GetInput());
  }

  GetOutput() = std::vector<int>();
}

bool TsibarevaEEdgeSelectSobelMPI::ValidationImpl() {
  return true;
}

bool TsibarevaEEdgeSelectSobelMPI::PreProcessingImpl() {
  return true;
}

bool TsibarevaEEdgeSelectSobelMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  BroadcastParameters();

  DistributeRows();

  std::vector<int> local_result = LocalGradientsComputing();

  GatherResults(local_result);

  return true;
}

bool TsibarevaEEdgeSelectSobelMPI::PostProcessingImpl() {
  return true;
}

void TsibarevaEEdgeSelectSobelMPI::BroadcastParameters() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&threshold_, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void TsibarevaEEdgeSelectSobelMPI::RowDistributionComputing(int world_rank, int world_size, int &base_rows,
                                                            int &remainder, int &real_rows, int &need_top_halo,
                                                            int &need_bottom_halo, int &total_rows) {
  // базовое (основное) количество строк на процесс
  base_rows = height_ / world_size;
  remainder = height_ % world_size;

  // предварительное количество строк на процесс
  real_rows = base_rows + (world_rank < remainder ? 1 : 0);
  local_height_ = real_rows;

  // отдельно подсчитаны флаги, какому процессу требуется верхняя соседняя строка, какому - нижняя соседняя строка
  need_top_halo = (world_rank > 0) ? 1 : 0;
  need_bottom_halo = (world_rank < (world_size - 1)) ? 1 : 0;

  // итоговое количество строк на процесс
  total_rows = real_rows + need_top_halo + need_bottom_halo;
  local_height_with_halo_ = total_rows;
  local_pixels_.resize(static_cast<size_t>(total_rows) * width_, 0);
}

void TsibarevaEEdgeSelectSobelMPI::SendParameters(int world_rank, int world_size, int base_rows, int remainder,
                                                  std::vector<int> &real_rows_per_proc, std::vector<int> &send_counts,
                                                  std::vector<int> &send_displs) const {
  if (world_rank == 0) {
    int current_row = 0;
    for (int dest = 0; dest < world_size; ++dest) {
      int dest_real_rows = base_rows + (dest < remainder ? 1 : 0);
      real_rows_per_proc[dest] = dest_real_rows;

      int dest_need_top_halo = (dest > 0) ? 1 : 0;
      int dest_need_bottom_halo = (dest < (world_size - 1)) ? 1 : 0;
      int start_row_with_halo = current_row - dest_need_top_halo;

      int end_row_with_halo = current_row + dest_real_rows + dest_need_bottom_halo - 1;
      end_row_with_halo = std::min(end_row_with_halo, height_ - 1);

      int actual_rows = end_row_with_halo - start_row_with_halo + 1;

      send_counts[dest] = actual_rows * width_;
      send_displs[dest] = start_row_with_halo * width_;

      current_row += dest_real_rows;
    }
  }
}

void TsibarevaEEdgeSelectSobelMPI::DataDistribution(int world_rank, const std::vector<int> &send_counts,
                                                    const std::vector<int> &send_displs) {
  MPI_Scatterv(world_rank == 0 ? input_pixels_.data() : nullptr, send_counts.data(), send_displs.data(), MPI_INT,
               local_pixels_.data(), static_cast<int>(local_pixels_.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

void TsibarevaEEdgeSelectSobelMPI::DistributeRows() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int base_rows = 0;
  int remainder = 0;
  int real_rows = 0;
  int need_top_halo = 0;
  int need_bottom_halo = 0;
  int total_rows = 0;

  RowDistributionComputing(world_rank, world_size, base_rows, remainder, real_rows, need_top_halo, need_bottom_halo,
                           total_rows);

  std::vector<int> send_counts(world_size, 0);
  std::vector<int> send_displs(world_size, 0);
  std::vector<int> real_rows_per_proc(world_size, 0);

  SendParameters(world_rank, world_size, base_rows, remainder, real_rows_per_proc, send_counts, send_displs);

  DataDistribution(world_rank, send_counts, send_displs);
}

std::vector<int> TsibarevaEEdgeSelectSobelMPI::LocalGradientsComputing() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<int> local_result;
  if (local_height_ > 0) {
    local_result.resize(static_cast<size_t>(local_height_) * width_, 0);

    for (int local_y = 0; local_y < local_height_; ++local_y) {
      int y = local_y + ((world_rank > 0) ? 1 : 0);

      for (int col = 0; col < width_; ++col) {
        int gx = GradientX(col, y);
        int gy = GradientY(col, y);

        int mag = static_cast<int>(std::sqrt((gx * gx) + (gy * gy) + 0.0));
        local_result[(static_cast<size_t>(local_y) * width_) + col] = (mag <= threshold_) ? 0 : mag;
      }
    }
  }

  return local_result;
}

int TsibarevaEEdgeSelectSobelMPI::GradientX(int x, int y) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < local_height_with_halo_) {
        int pixel = local_pixels_[(static_cast<size_t>(ny) * width_) + nx];
        sum += pixel * kSobelX[ky + 1][kx + 1];
      }
    }
  }

  return sum;
}

int TsibarevaEEdgeSelectSobelMPI::GradientY(int x, int y) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < local_height_with_halo_) {
        int pixel = local_pixels_[(static_cast<size_t>(ny) * width_) + nx];
        sum += pixel * kSobelY[ky + 1][kx + 1];
      }
    }
  }

  return sum;
}

void TsibarevaEEdgeSelectSobelMPI::GatherResults(const std::vector<int> &local_result) {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<int> local_result_sizes(world_size);
  int local_size = static_cast<int>(local_result.size());
  MPI_Allgather(&local_size, 1, MPI_INT, local_result_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displs(world_size);
  int total_size = 0;
  for (int i = 0; i < world_size; ++i) {
    displs[i] = total_size;
    total_size += local_result_sizes[i];
  }

  GetOutput().resize(static_cast<size_t>(total_size));

  MPI_Allgatherv(local_result.empty() ? nullptr : local_result.data(), local_size, MPI_INT, GetOutput().data(),
                 local_result_sizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
}

}  // namespace tsibareva_e_edge_select_sobel
