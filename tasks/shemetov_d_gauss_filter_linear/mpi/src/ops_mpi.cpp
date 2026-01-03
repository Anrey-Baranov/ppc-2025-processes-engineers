#include "shemetov_d_gauss_filter_linear/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"

namespace shemetov_d_gauss_filter_linear {

GaussFilterMPI::GaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

Pixel GaussFilterMPI::ApplyKernel(const InType &in, int i, int j, const std::vector<std::vector<float>> &kernel) {
  float channel_red = 0.F;
  float channel_green = 0.F;
  float channel_blue = 0.F;

  for (int ki = -1; ki <= 1; ++ki) {
    for (int kj = -1; kj <= 1; ++kj) {
      const auto &lnk_pixel = in[i + ki][j + kj];
      float coefficient = kernel[ki + 1][kj + 1];

      channel_red += coefficient * static_cast<float>(lnk_pixel.channel_red);
      channel_green += coefficient * static_cast<float>(lnk_pixel.channel_green);
      channel_blue += coefficient * static_cast<float>(lnk_pixel.channel_blue);
    }
  }

  Pixel m_pixel = {.channel_red = static_cast<uint8_t>(std::clamp(channel_red, 0.F, 255.F)),
                   .channel_green = static_cast<uint8_t>(std::clamp(channel_green, 0.F, 255.F)),
                   .channel_blue = static_cast<uint8_t>(std::clamp(channel_blue, 0.F, 255.F))};
  return m_pixel;
}

void GaussFilterMPI::ComputeLocalBlock(const InType &in, int start_row, std::vector<std::vector<Pixel>> &local_out) {
  const std::vector<std::vector<float>> kernel = {
      {1.F / 16, 2.F / 16, 1.F / 16}, {2.F / 16, 4.F / 16, 2.F / 16}, {1.F / 16, 2.F / 16, 1.F / 16}};

  for (int i = 0; i < local_rows; ++i) {
    const int global_row = start_row + i;

    if (global_row == 0 || global_row == height - 1) {
      continue;
    }

    for (int j = 1; j < width - 1; ++j) {
      local_out[static_cast<size_t>(i)][static_cast<size_t>(j)] = ApplyKernel(in, global_row, j, kernel);
    }
  }
}

std::vector<uint8_t> GaussFilterMPI::SendColumns(const std::vector<std::vector<Pixel>> &local_out, size_t column) {
  const auto sc_size = static_cast<size_t>(local_rows) * 3;
  std::vector<uint8_t> send_columns(sc_size);

  for (size_t i = 0; std::cmp_less(i, local_rows); ++i) {
    send_columns[(i * 3)] = local_out[i][column].channel_red;
    send_columns[(i * 3) + 1] = local_out[i][column].channel_green;
    send_columns[(i * 3) + 2] = local_out[i][column].channel_blue;
  }

  return send_columns;
}

void GaussFilterMPI::RecieveColumns(std::vector<uint8_t> &recieve_columns, size_t column,
                                    std::vector<std::vector<Pixel>> &out) {
  for (size_t i = 0; std::cmp_less(i, height); ++i) {
    out[i][column].channel_red = recieve_columns[(i * 3)];
    out[i][column].channel_green = recieve_columns[(i * 3) + 1];
    out[i][column].channel_blue = recieve_columns[(i * 3) + 2];
  }
}

void GaussFilterMPI::GatherResult(const std::vector<std::vector<Pixel>> &local_out, const InType &in,
                                  std::vector<std::vector<Pixel>> &out) {
  std::vector<int> string_count(size, 0);
  for (int rank_idx = 0; rank_idx < size; ++rank_idx) {
    int distribution = rank_idx < extra_rows ? 1 : 0;
    string_count[rank_idx] = (base_rows + distribution) * 3;
  }

  std::vector<int> displacement(size, 0);
  for (int rank_idx = 1; rank_idx < size; ++rank_idx) {
    displacement[rank_idx] = displacement[rank_idx - 1] + string_count[rank_idx - 1];
  }

  for (size_t j = 0; std::cmp_less(j, width); ++j) {
    auto send_columns = SendColumns(local_out, j);

    if (rank == 0) {
      const auto rc_size = static_cast<size_t>(height) * 3;
      std::vector<uint8_t> recieve_columns(rc_size);

      MPI_Gatherv(send_columns.data(), local_rows * 3, MPI_UNSIGNED_CHAR, recieve_columns.data(), string_count.data(),
                  displacement.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

      RecieveColumns(recieve_columns, j, out);
    } else {
      MPI_Gatherv(send_columns.data(), local_rows * 3, MPI_UNSIGNED_CHAR, nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
                  0, MPI_COMM_WORLD);
    }
  }

  if (rank == 0) {
    for (size_t i = 0; std::cmp_less(i, height); ++i) {
      out[i][0] = in[i][0];
      out[i][width - 1] = in[i][width - 1];
    }
    for (size_t j = 0; std::cmp_less(j, width); ++j) {
      out[0][j] = in[0][j];
      out[height - 1][j] = in[height - 1][j];
    }
  }
}

bool GaussFilterMPI::ValidationImpl() {
  const auto &in = GetInput();
  return !in.empty() && !in[0].empty();
}

bool GaussFilterMPI::PreProcessingImpl() {
  return true;
}

bool GaussFilterMPI::RunImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();
  auto &out = GetOutput();

  height = static_cast<int>(in.size());
  width = static_cast<int>(in[0].size());

  if (height < 3 || width < 3) {
    out = in;
    return true;
  }

  base_rows = height / size;
  extra_rows = height % size;

  local_rows = base_rows + ((rank < extra_rows) ? 1 : 0);
  std::vector<std::vector<Pixel>> local_out(static_cast<size_t>(local_rows),
                                            std::vector<Pixel>(static_cast<size_t>(width)));

  const int start_row = (rank * base_rows) + std::min(rank, extra_rows);

  ComputeLocalBlock(in, start_row, local_out);

  GatherResult(local_out, in, out);

  return true;
}

bool GaussFilterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_gauss_filter_linear
