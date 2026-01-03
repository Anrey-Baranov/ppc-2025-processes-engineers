#include "rastvorov_k_radix_sort_double_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rastvorov_k_radix_sort_double_merge {

namespace {

inline uint64_t DoubleToOrderedKey(double x) {
  if (std::isnan(x)) {
    return UINT64_MAX;
  }
  const auto bits = std::bit_cast<uint64_t>(x);
  const uint64_t sign = bits >> 63U;
  if (sign != 0U) {
    return ~bits;
  }
  return bits ^ (1ULL << 63U);
}

void RadixSortDouble(std::vector<double> *vec) {
  auto &a = *vec;
  const std::size_t n = a.size();
  if (n <= 1) {
    return;
  }

  std::vector<double> out(n);
  std::vector<uint64_t> keys(n);
  std::vector<uint64_t> out_keys(n);

  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = DoubleToOrderedKey(a[i]);
  }

  for (std::size_t pass = 0; pass < 8; ++pass) {
    std::array<std::size_t, 256> count{};
    const std::size_t shift = pass * 8;

    for (std::size_t i = 0; i < n; ++i) {
      const auto byte = static_cast<unsigned>((keys[i] >> shift) & 0xFFULL);
      count.at(static_cast<std::size_t>(byte))++;
    }

    std::array<std::size_t, 256> pos{};
    pos.at(0) = 0;
    for (std::size_t byte_idx = 1; byte_idx < pos.size(); ++byte_idx) {
      pos.at(byte_idx) = pos.at(byte_idx - 1) + count.at(byte_idx - 1);
    }

    for (std::size_t i = 0; i < n; ++i) {
      const auto byte = static_cast<unsigned>((keys[i] >> shift) & 0xFFULL);
      const std::size_t p = pos.at(static_cast<std::size_t>(byte))++;
      out[p] = a[i];
      out_keys[p] = keys[i];
    }

    a.swap(out);
    keys.swap(out_keys);
  }
}

std::vector<double> MergeSorted(const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> out;
  out.reserve(a.size() + b.size());

  std::size_t i = 0;
  std::size_t j = 0;

  while (i < a.size() && j < b.size()) {
    const uint64_t ka = DoubleToOrderedKey(a[i]);
    const uint64_t kb = DoubleToOrderedKey(b[j]);
    if (ka <= kb) {
      out.push_back(a[i++]);
    } else {
      out.push_back(b[j++]);
    }
  }

  while (i < a.size()) {
    out.push_back(a[i++]);
  }
  while (j < b.size()) {
    out.push_back(b[j++]);
  }

  return out;
}

std::vector<double> RecvVectorD(int src, int tag_base, MPI_Comm comm) {
  int sz = 0;
  MPI_Status status{};
  MPI_Recv(&sz, 1, MPI_INT, src, tag_base, comm, &status);

  std::vector<double> v(static_cast<std::size_t>(sz));
  if (sz > 0) {
    MPI_Recv(v.data(), sz, MPI_DOUBLE, src, tag_base + 1, comm, &status);
  }
  return v;
}

void SendVectorD(int dst, int tag_base, const std::vector<double> &v, MPI_Comm comm) {
  const int sz = static_cast<int>(v.size());
  MPI_Send(&sz, 1, MPI_INT, dst, tag_base, comm);
  if (sz > 0) {
    MPI_Send(v.data(), sz, MPI_DOUBLE, dst, tag_base + 1, comm);
  }
}

}  // namespace

bool RastvorovKRadixSortDoubleMergeMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  return true;
}

bool RastvorovKRadixSortDoubleMergeMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int global_size = 0;
  if (world_rank_ == 0) {
    global_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  counts_.assign(world_size_, 0);
  displs_.assign(world_size_, 0);

  const int base = (world_size_ > 0) ? (global_size / world_size_) : 0;
  const int rem = (world_size_ > 0) ? (global_size % world_size_) : 0;

  for (int rank_idx = 0; rank_idx < world_size_; ++rank_idx) {
    counts_[rank_idx] = base + ((rank_idx < rem) ? 1 : 0);
  }
  for (int rank_idx = 1; rank_idx < world_size_; ++rank_idx) {
    displs_[rank_idx] = displs_[rank_idx - 1] + counts_[rank_idx - 1];
  }

  local_.assign(static_cast<std::size_t>(counts_[world_rank_]), 0.0);

  double *send_buf = nullptr;
  if (world_rank_ == 0 && global_size > 0) {
    send_buf = GetInput().data();
  }

  MPI_Scatterv(send_buf, counts_.data(), displs_.data(), MPI_DOUBLE, local_.data(), counts_[world_rank_], MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  GetOutput().clear();
  return true;
}

bool RastvorovKRadixSortDoubleMergeMPI::RunImpl() {
  RadixSortDouble(&local_);

  for (int step = 1; step < world_size_; step <<= 1) {
    if ((world_rank_ % (2 * step)) == 0) {
      const int partner = world_rank_ + step;
      if (partner < world_size_) {
        std::vector<double> other = RecvVectorD(partner, 2000 + step, MPI_COMM_WORLD);
        local_ = MergeSorted(local_, other);
      }
    } else {
      const int partner = world_rank_ - step;
      SendVectorD(partner, 2000 + step, local_, MPI_COMM_WORLD);
      break;
    }
  }
  return true;
}

bool RastvorovKRadixSortDoubleMergeMPI::PostProcessingImpl() {
  int out_size = 0;
  if (world_rank_ == 0) {
    GetOutput() = local_;
    out_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank_ != 0) {
    GetOutput().assign(static_cast<std::size_t>(out_size), 0.0);
  }

  if (out_size > 0) {
    MPI_Bcast(GetOutput().data(), out_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  return true;
}

}  // namespace rastvorov_k_radix_sort_double_merge
