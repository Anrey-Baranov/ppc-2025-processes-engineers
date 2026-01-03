#include "pankov_gauss_filter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "pankov_gauss_filter/common/include/common.hpp"

namespace pankov_gauss_filter {

namespace {

inline int ClampInt(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

inline std::uint8_t ClampToByte(int v) {
  v = std::max(0, std::min(v, 255));
  return static_cast<std::uint8_t>(v);
}

constexpr std::array<std::array<int, 3>, 3> kGaussianKernel3x3 = {{{{1, 2, 1}}, {{2, 4, 2}}, {{1, 2, 1}}}};
constexpr int kGaussianDiv = 16;

struct Decomposition {
  std::size_t width = 0;
  std::size_t base_cols = 0;
  std::size_t rem_cols = 0;

  [[nodiscard]] std::size_t StartColForProc(int proc_rank) const {
    return (static_cast<std::size_t>(proc_rank) * base_cols) +
           static_cast<std::size_t>(std::min(proc_rank, static_cast<int>(rem_cols)));
  }

  [[nodiscard]] std::size_t LocalWidthForProc(int proc_rank) const {
    return base_cols + (std::cmp_less(proc_rank, static_cast<int>(rem_cols)) ? 1U : 0U);
  }
};

Decomposition MakeDecomposition(std::size_t width, int proc_count) {
  Decomposition dec;
  dec.width = width;
  dec.base_cols = width / static_cast<std::size_t>(proc_count);
  dec.rem_cols = width % static_cast<std::size_t>(proc_count);
  return dec;
}

std::pair<std::vector<int>, std::vector<int>> BuildSendCountsDispls(std::size_t height, std::size_t channels,
                                                                    const Decomposition &dec, int proc_count) {
  std::vector<int> sendcounts(static_cast<std::size_t>(proc_count), 0);
  std::vector<int> displs(static_cast<std::size_t>(proc_count), 0);

  int disp = 0;
  for (int proc_rank = 0; proc_rank < proc_count; ++proc_rank) {
    const std::size_t local_width = dec.LocalWidthForProc(proc_rank);
    const std::size_t cnt = height * local_width * channels;
    sendcounts[static_cast<std::size_t>(proc_rank)] = static_cast<int>(cnt);
    displs[static_cast<std::size_t>(proc_rank)] = disp;
    disp += static_cast<int>(cnt);
  }

  return {sendcounts, displs};
}

void PackStripeForProc(const Image &image, const Decomposition &dec, std::size_t height, std::size_t channels,
                       int proc_rank, std::vector<std::uint8_t> *packed) {
  const std::size_t local_width = dec.LocalWidthForProc(proc_rank);
  const std::size_t start_col = dec.StartColForProc(proc_rank);
  packed->assign(height * local_width * channels, 0);

  if (local_width == 0) {
    return;
  }

  for (std::size_t row_idx = 0; row_idx < height; ++row_idx) {
    for (std::size_t local_col_idx = 0; local_col_idx < local_width; ++local_col_idx) {
      const std::size_t src_col = start_col + local_col_idx;
      const std::size_t src_idx = (row_idx * dec.width + src_col) * channels;
      const std::size_t dst_idx = (row_idx * local_width + local_col_idx) * channels;
      for (std::size_t ch_idx = 0; ch_idx < channels; ++ch_idx) {
        (*packed)[dst_idx + ch_idx] = image.data[src_idx + ch_idx];
      }
    }
  }
}

void DistributeStripes(const Image &image, const Decomposition &dec, std::size_t height, std::size_t channels,
                       int proc_rank, int proc_count, std::vector<std::uint8_t> *local_stripe) {
  const std::size_t local_width = dec.LocalWidthForProc(proc_rank);
  const std::size_t local_elems = height * local_width * channels;
  local_stripe->assign(local_elems, 0);

  if (proc_rank == 0) {
    PackStripeForProc(image, dec, height, channels, 0, local_stripe);
    std::vector<std::uint8_t> tmp;
    for (int other_rank = 1; other_rank < proc_count; ++other_rank) {
      PackStripeForProc(image, dec, height, channels, other_rank, &tmp);
      MPI_Send(tmp.data(), static_cast<int>(tmp.size()), MPI_UNSIGNED_CHAR, other_rank, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_stripe->data(), static_cast<int>(local_elems), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void ExtractStripeColumn(const std::vector<std::uint8_t> &stripe, std::size_t stripe_width, std::size_t height,
                         std::size_t channels, std::size_t col_idx, std::vector<std::uint8_t> *column) {
  column->assign(height * channels, 0);
  for (std::size_t row_idx = 0; row_idx < height; ++row_idx) {
    const std::size_t base = (row_idx * stripe_width + col_idx) * channels;
    for (std::size_t ch_idx = 0; ch_idx < channels; ++ch_idx) {
      (*column)[(row_idx * channels) + ch_idx] = stripe[base + ch_idx];
    }
  }
}

void ExchangeHaloColumns(const Decomposition &dec, std::size_t height, std::size_t channels, int proc_rank,
                         int proc_count, const std::vector<std::uint8_t> &local_stripe, std::size_t local_width,
                         std::vector<std::uint8_t> *left_halo, std::vector<std::uint8_t> *right_halo) {
  const std::size_t col_elems = height * channels;
  left_halo->assign(col_elems, 0);
  right_halo->assign(col_elems, 0);

  if (local_width == 0) {
    return;
  }

  std::vector<std::uint8_t> first_col;
  std::vector<std::uint8_t> last_col;
  ExtractStripeColumn(local_stripe, local_width, height, channels, 0, &first_col);
  ExtractStripeColumn(local_stripe, local_width, height, channels, local_width - 1, &last_col);

  *left_halo = first_col;  // replicate by default
  *right_halo = last_col;  // replicate by default

  const bool has_left_neighbor = (proc_rank > 0) && (dec.LocalWidthForProc(proc_rank - 1) > 0) && (local_width > 0);
  const bool has_right_neighbor =
      (proc_rank + 1 < proc_count) && (dec.LocalWidthForProc(proc_rank + 1) > 0) && (local_width > 0);

  if (has_left_neighbor) {
    MPI_Sendrecv(first_col.data(), static_cast<int>(col_elems), MPI_UNSIGNED_CHAR, proc_rank - 1, 100,
                 left_halo->data(), static_cast<int>(col_elems), MPI_UNSIGNED_CHAR, proc_rank - 1, 200, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
  }
  if (has_right_neighbor) {
    MPI_Sendrecv(last_col.data(), static_cast<int>(col_elems), MPI_UNSIGNED_CHAR, proc_rank + 1, 200,
                 right_halo->data(), static_cast<int>(col_elems), MPI_UNSIGNED_CHAR, proc_rank + 1, 100, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
  }
}

std::uint8_t ConvolveGaussian3x3ForStripe(const std::vector<std::uint8_t> &local_stripe, std::size_t local_width,
                                          const std::vector<std::uint8_t> &left_halo,
                                          const std::vector<std::uint8_t> &right_halo, int height, int channels,
                                          int row, std::size_t local_col, int channel) {
  const auto u_channels = static_cast<std::size_t>(channels);

  const int row_prev = ClampInt(row - 1, 0, height - 1);
  const int row_curr = row;
  const int row_next = ClampInt(row + 1, 0, height - 1);
  const std::array<int, 3> src_rows = {row_prev, row_curr, row_next};

  int acc = 0;
  for (int dy_idx = 0; dy_idx < 3; ++dy_idx) {
    const int src_row = src_rows.at(static_cast<std::size_t>(dy_idx));
    const std::size_t halo_row_off = static_cast<std::size_t>(src_row) * u_channels;
    for (int dx_idx = 0; dx_idx < 3; ++dx_idx) {
      const int weight = kGaussianKernel3x3.at(static_cast<std::size_t>(dy_idx)).at(static_cast<std::size_t>(dx_idx));
      const int offset = dx_idx - 1;

      std::uint8_t pix = 0;
      if (offset == -1 && local_col == 0) {
        pix = left_halo[halo_row_off + static_cast<std::size_t>(channel)];
      } else if (offset == 1 && (local_col + 1 == local_width)) {
        pix = right_halo[halo_row_off + static_cast<std::size_t>(channel)];
      } else {
        std::size_t src_local_col = local_col;
        if (offset == -1) {
          src_local_col = local_col - 1;
        } else if (offset == 1) {
          src_local_col = local_col + 1;
        }
        const std::size_t idx = ((static_cast<std::size_t>(src_row) * local_width + src_local_col) * u_channels) +
                                static_cast<std::size_t>(channel);
        pix = local_stripe[idx];
      }
      acc += weight * static_cast<int>(pix);
    }
  }

  const int rounded = (acc + (kGaussianDiv / 2)) / kGaussianDiv;
  return ClampToByte(rounded);
}

void ApplyGaussianToStripe(const std::vector<std::uint8_t> &local_stripe, std::size_t local_width,
                           const std::vector<std::uint8_t> &left_halo, const std::vector<std::uint8_t> &right_halo,
                           int height, int channels, std::vector<std::uint8_t> *local_out) {
  local_out->assign(static_cast<std::size_t>(height) * local_width * static_cast<std::size_t>(channels), 0);

  for (int row = 0; row < height; ++row) {
    for (std::size_t local_col = 0; local_col < local_width; ++local_col) {
      for (int channel = 0; channel < channels; ++channel) {
        const std::size_t out_idx =
            ((static_cast<std::size_t>(row) * local_width + local_col) * static_cast<std::size_t>(channels)) +
            static_cast<std::size_t>(channel);
        (*local_out)[out_idx] = ConvolveGaussian3x3ForStripe(local_stripe, local_width, left_halo, right_halo, height,
                                                             channels, row, local_col, channel);
      }
    }
  }
}

void UnpackGatheredToImage(const Decomposition &dec, std::size_t height, std::size_t channels, int proc_count,
                           const std::vector<std::uint8_t> &gathered, const std::vector<int> &displs,
                           std::vector<std::uint8_t> *global_out) {
  global_out->assign(dec.width * height * channels, 0);
  for (int proc_rank = 0; proc_rank < proc_count; ++proc_rank) {
    const std::size_t local_width = dec.LocalWidthForProc(proc_rank);
    if (local_width == 0) {
      continue;
    }
    const std::size_t start_col = dec.StartColForProc(proc_rank);
    const auto offset = static_cast<std::size_t>(displs[static_cast<std::size_t>(proc_rank)]);
    for (std::size_t row_idx = 0; row_idx < height; ++row_idx) {
      for (std::size_t local_col_idx = 0; local_col_idx < local_width; ++local_col_idx) {
        const std::size_t dst_col = start_col + local_col_idx;
        const std::size_t dst_idx = (row_idx * dec.width + dst_col) * channels;
        const std::size_t src_idx = offset + ((row_idx * local_width + local_col_idx) * channels);
        for (std::size_t ch_idx = 0; ch_idx < channels; ++ch_idx) {
          (*global_out)[dst_idx + ch_idx] = gathered[src_idx + ch_idx];
        }
      }
    }
  }
}

}  // namespace

PankovGaussFilterMPI::PankovGaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp(in);
  std::swap(GetInput(), temp);
  GetOutput() = OutType{};
}

bool PankovGaussFilterMPI::ValidationImpl() {
  const auto &in = GetInput();
  if (!GetOutput().data.empty()) {
    return false;
  }
  if (in.width < 0 || in.height < 0 || in.channels < 0) {
    return false;
  }
  if (in.width == 0 || in.height == 0) {
    return in.data.empty();
  }
  if (in.channels == 0) {
    return false;
  }
  const auto expected =
      static_cast<std::size_t>(in.width) * static_cast<std::size_t>(in.height) * static_cast<std::size_t>(in.channels);
  return in.data.size() == expected;
}

bool PankovGaussFilterMPI::PreProcessingImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  out.width = in.width;
  out.height = in.height;
  out.channels = in.channels;
  const auto total =
      static_cast<std::size_t>(in.width) * static_cast<std::size_t>(in.height) * static_cast<std::size_t>(in.channels);
  out.data.assign(total, 0);
  return true;
}

bool PankovGaussFilterMPI::RunImpl() {
  int proc_rank = 0;
  int proc_count = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  const auto &in = GetInput();
  int width = 0;
  int height = 0;
  int channels = 0;

  if (proc_rank == 0) {
    width = in.width;
    height = in.height;
    channels = in.channels;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (width == 0 || height == 0) {
    auto &out = GetOutput();
    out.width = width;
    out.height = height;
    out.channels = channels;
    out.data.clear();
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  const auto u_w = static_cast<std::size_t>(width);
  const auto u_h = static_cast<std::size_t>(height);
  const auto u_ch = static_cast<std::size_t>(channels);

  const Decomposition dec = MakeDecomposition(u_w, proc_count);
  const std::size_t local_width = dec.LocalWidthForProc(proc_rank);
  const std::size_t local_elems = u_h * local_width * u_ch;

  std::vector<int> sendcounts(static_cast<std::size_t>(proc_count), 0);
  std::vector<int> displs(static_cast<std::size_t>(proc_count), 0);
  if (proc_rank == 0) {
    auto counts_displs = BuildSendCountsDispls(u_h, u_ch, dec, proc_count);
    sendcounts = std::move(counts_displs.first);
    displs = std::move(counts_displs.second);
  }

  std::vector<std::uint8_t> local_stripe;
  DistributeStripes(in, dec, u_h, u_ch, proc_rank, proc_count, &local_stripe);

  std::vector<std::uint8_t> left_halo;
  std::vector<std::uint8_t> right_halo;
  ExchangeHaloColumns(dec, u_h, u_ch, proc_rank, proc_count, local_stripe, local_width, &left_halo, &right_halo);

  std::vector<std::uint8_t> local_out;
  ApplyGaussianToStripe(local_stripe, local_width, left_halo, right_halo, height, channels, &local_out);

  const std::size_t total = u_w * u_h * u_ch;
  std::vector<std::uint8_t> gathered;
  if (proc_rank == 0) {
    gathered.resize(total);
  }

  std::uint8_t *recv_buf = (proc_rank == 0) ? gathered.data() : nullptr;
  int *recv_counts = (proc_rank == 0) ? sendcounts.data() : nullptr;
  int *recv_displs = (proc_rank == 0) ? displs.data() : nullptr;

  MPI_Gatherv(local_out.data(), static_cast<int>(local_elems), MPI_UNSIGNED_CHAR, recv_buf, recv_counts, recv_displs,
              MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  std::vector<std::uint8_t> global_out;
  if (proc_rank == 0) {
    UnpackGatheredToImage(dec, u_h, u_ch, proc_count, gathered, displs, &global_out);
  } else {
    global_out.assign(total, 0);
  }

  MPI_Bcast(global_out.data(), static_cast<int>(global_out.size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  auto &out = GetOutput();
  out.width = width;
  out.height = height;
  out.channels = channels;
  out.data = std::move(global_out);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool PankovGaussFilterMPI::PostProcessingImpl() {
  const auto &out = GetOutput();
  if (out.width == 0 || out.height == 0) {
    return out.data.empty();
  }
  return !out.data.empty();
}

}  // namespace pankov_gauss_filter
